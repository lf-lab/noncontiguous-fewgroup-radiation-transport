"""
fewgroup.py

Library for the non-contiguous generalized-group ("few-group") model.

Idea
    The fine-group spectral energy density u_g(x) is expanded with K modes
    whose supports S_k (sets of fine groups) do not overlap:

        u_g(x) ~ a_k(x) w_k(g)   for g in S_k,     sum_{g in S_k} w_k(g)^2 dE_g = 1.

    S_k does NOT have to be a contiguous band of photon energy.  Because the
    supports are disjoint, the Galerkin-projected equations are K independent
    scalar diffusion equations (no coupling between the modes):

        -d/dx[ D_k da_k/dx ] + c kappa_bar_k a_k = eps_bar_k
        kappa_bar_k = sum w_k^2 kappa_g dE_g            (absorption)
        eps_bar_k   = sum w_k   eps_g   dE_g            (emission)
        kappa_D,k   = 1 / sum (w_k^2 dE_g / kappa_g)    (diffusion; D = c lambda(R)/kappa_D)

    These three quantities are tabulated on the (T_e, n_i) grid of the fine
    table.  The result is an ordinary K-group table and it is solved by an
    ordinary multigroup diffusion solver.

How S_k and w_k are found (only the Gram matrix of training spectra is used)
    1. Weighted kernel k-means (cosine similarity between the fine groups,
       k-means++ start) gives a first partition into K generalized groups.
    2. For each group, the first POD mode of its block is the shape w_k
       (non-negative).  Each fine group is then moved to the mode that
       explains its energy best (Lloyd iteration with rank-one blocks).
"""
import time
import numpy as np
from scipy.interpolate import RectBivariateSpline

import mg_diffusion as M


# --------------------------------------------------------------------------
# Partition of the fine groups into generalized groups
# --------------------------------------------------------------------------
def gram_to_A(gram, n_samples, dE):
    """dE-weighted Gram matrix  A = W^(1/2) (U^T U / n) W^(1/2)."""
    s = np.sqrt(dE)
    A = s[:, None] * (gram / n_samples) * s[None, :]
    return 0.5 * (A + A.T)


def kernel_kmeans(Cs, m, K, rng, n_iter=100):
    """Weighted kernel k-means.  Cs: (G, G) cosine kernel, m: (G,) weights.
    Returns the label (G,) of each fine group."""
    G = Cs.shape[0]
    centers = [rng.choice(G, p=m / m.sum())]                   # k-means++ initialization
    d2 = 2 - 2 * Cs[:, centers[0]]
    for _ in range(1, K):
        p = m * np.maximum(d2, 0)
        p /= p.sum()
        c = rng.choice(G, p=p)
        centers.append(c)
        d2 = np.minimum(d2, 2 - 2 * Cs[:, c])
    lab = np.argmin(2 - 2 * Cs[:, centers], axis=1)
    dmin = np.min(2 - 2 * Cs[:, centers], axis=1)
    for _ in range(n_iter):
        Z = np.zeros((G, K))
        Z[np.arange(G), lab] = 1
        Zm = Z * m[:, None]
        Mk = Zm.sum(axis=0)
        empty = Mk == 0
        if empty.any():                                        # empty cluster: give it the farthest point
            for k in np.flatnonzero(empty):
                g = np.argmax(m * dmin)
                lab[g] = k
                dmin[g] = 0.0
            continue
        t2 = (Cs @ Zm) / Mk[None, :]
        t3 = np.einsum('gk,gk->k', Zm, Cs @ Zm) / Mk ** 2
        d = 1 - 2 * t2 + t3[None, :]
        new = np.argmin(d, axis=1)
        dmin = d[np.arange(G), new]
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def block_modes(A, dE, lab, K):
    """First POD mode of each generalized group.
    Returns modes (K, G) = w_k and sig2 (K,) = energy explained by each mode."""
    G = len(dE)
    modes = np.zeros((K, G))
    sig2 = np.zeros(K)
    for k in range(K):
        S = np.flatnonzero(lab == k)
        if S.size == 0:
            continue
        lam, V = np.linalg.eigh(A[np.ix_(S, S)])
        v = V[:, -1]
        v = v * np.sign(v.sum() + 1e-300)
        v = np.maximum(v, 0)
        v /= np.linalg.norm(v) + 1e-300
        modes[k, S] = v / np.sqrt(dE[S])
        sig2[k] = lam[-1]
    return modes, sig2


def refine_partition(A, dE, lab, K, n_iter=10):
    """Lloyd iteration with rank-one blocks: move each fine group to the mode
    that explains its energy A_gg best."""
    G = len(dE)
    Agg = np.diag(A)
    for _ in range(n_iter):
        modes, sig2 = block_modes(A, dE, lab, K)
        V = modes * np.sqrt(dE)[None, :]
        proj = A @ V.T                                         # (G, K)
        expl = proj ** 2 / np.maximum(sig2, 1e-300)[None, :]   # explained energy
        new = np.argmax(expl, axis=1)
        for k in range(K):                                     # refill empty groups
            if not np.any(new == k):
                g = np.argmin(expl[np.arange(G), new] / Agg)
                new[g] = k
        if np.array_equal(new, lab):
            break
        lab = new
    modes, sig2 = block_modes(A, dE, lab, K)
    return lab, modes, sig2.sum() / np.trace(A)


def noncontiguous_partition(A, dE, K, rng):
    """Proposed method.  Returns label (G,), modes (K, G), explained energy fraction."""
    Agg = np.diag(A)
    Cs = A / np.sqrt(np.outer(Agg, Agg))
    lab = kernel_kmeans(Cs, Agg, K, rng)
    return refine_partition(A, dE, lab, K)


def conventional_partition(A, dE, K):
    """Reference: conventional multigroup = contiguous bands + flat shape.
    The band edges are quantiles of the training energy."""
    c = np.cumsum(np.diag(A))
    c /= c[-1]
    lab = np.minimum((c * K).astype(int), K - 1)
    _, lab = np.unique(lab, return_inverse=True)               # remove empty bands
    K = int(lab.max()) + 1
    modes = np.zeros((K, len(dE)))
    for k in range(K):
        S = np.flatnonzero(lab == k)
        modes[k, S] = 1.0 / np.sqrt(dE[S].sum())
    V = modes * np.sqrt(dE)[None, :]
    frac = np.einsum('kg,gh,kh->', V, A, V) / np.trace(A)
    return lab, modes, frac


# --------------------------------------------------------------------------
# K-group table
# --------------------------------------------------------------------------
def build_table(tab, modes, kappa_min=1e-5):
    """Project the fine table (M.OpacityTable) onto the modes.
    Returns log10 tables lkab, leps, lkD with shape (nTe, nNi, K)."""
    K, G = modes.shape
    w2 = modes ** 2 * tab.dE[None, :]
    w1 = modes * tab.dE[None, :]
    nT, nN = tab.lkap.shape[:2]
    kap = np.maximum(10.0 ** tab.lkap.reshape(nT * nN, G), kappa_min)
    eps = 10.0 ** tab.leps.reshape(nT * nN, G)
    lkab = np.log10(kap @ w2.T).reshape(nT, nN, K)
    leps = np.log10(np.maximum(eps @ w1.T, 1e-300)).reshape(nT, nN, K)
    lkD = np.log10(1.0 / ((1.0 / kap) @ w2.T)).reshape(nT, nN, K)
    return lkab, leps, lkD


def save_table(path, tab, lab, modes, frac, kappa_min=1e-5, **extra):
    lkab, leps, lkD = build_table(tab, modes, kappa_min)
    np.savez(path, K=modes.shape[0], E=tab.E, dE=tab.dE, label=lab, modes=modes,
             lTe=tab.lTe, lNi=tab.lNi, lkab=lkab, leps=leps, lkD=lkD,
             Zbar=tab.Zbar, A=tab.A, explained=frac, kappa_min=kappa_min, **extra)


def cubic_weights(xg, x, a=-0.5):
    """Indices and weights of local cubic (Catmull-Rom) interpolation on a uniform grid xg."""
    x = np.atleast_1d(np.asarray(x, float))
    h = xg[1] - xg[0]
    n = len(xg)
    f = (x - xg[0]) / h
    i = np.clip(np.floor(f).astype(int), 0, n - 2)
    s = np.clip(f - i, 0.0, 1.0)

    def k(t):
        t = np.abs(t)
        return np.where(t <= 1, (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1,
                        np.where(t < 2, a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a, 0.0))
    w = np.stack([k(s + 1), k(s), k(1 - s), k(2 - s)], axis=-1)
    idx = np.clip(np.stack([i - 1, i, i + 1, i + 2], axis=-1), 0, n - 1)
    return idx, w


class FewGroupModel:
    """Transport calculation that uses only the K-group table (.npz)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.K = int(z['K'])
        self.E, self.dE = z['E'], z['dE']
        self.label, self.modes = z['label'], z['modes']
        self.lTe, self.lNi = z['lTe'], z['lNi']
        self.Te, self.Ni = 10.0 ** self.lTe, 10.0 ** self.lNi
        self.lkab, self.leps, self.lkD = z['lkab'], z['leps'], z['lkD']
        self.A = float(z['A'])
        self.explained = float(z['explained'])
        self._zbar = RectBivariateSpline(self.lTe, self.lNi, z['Zbar'], kx=3, ky=3)

    def zbar(self, Te, ni):
        lTe = np.clip(np.log10(Te), self.lTe[0], self.lTe[-1])
        lni = np.clip(np.log10(ni), self.lNi[0], self.lNi[-1])
        return self._zbar(lTe, lni, grid=False)

    def interp(self, Te, ni):
        """kappa_bar, eps_bar, kappa_D at (Te, ni[:]); each (N, K)."""
        iT, wT = cubic_weights(self.lTe, np.log10(Te))
        iT, wT = iT[0], wT[0]
        iN, wN = cubic_weights(self.lNi, np.log10(ni))
        out = []
        for L in (self.lkab, self.leps, self.lkD):
            LT = np.tensordot(wT, L[iT], axes=(0, 0))          # (nNi, K)
            out.append(10.0 ** np.einsum('nj,njk->nk', wN, LT[iN]))
        return out

    def run(self, lambda_L, n_i, T_e, t_L, delta_x=1.0, x_max=500.0, left_bc='vacuum',
            limiter='larsen', n_larsen=2, verbose=False):
        """Same plasma, boundary conditions and flux limiter as mg_diffusion.run."""
        t0 = time.time()
        pl = M.build_plasma(self, lambda_L, n_i, T_e, t_L, x_max, delta_x, verbose)
        N_eff = int(np.sum(pl['ni'] >= self.Ni[0]))
        kab, eps, kD = self.interp(T_e, pl['ni'][:N_eff])
        a, F_k, info = M.solve_diffusion(kab, kD, eps, pl['dx'], left_bc, limiter, n_larsen)
        F_E = (F_k[:, None] * self.modes).sum(axis=0)          # fine-mesh spectrum [erg/cm^2/s/eV]
        info['time'] = time.time() - t0
        return dict(E_eV=self.E, dE=self.dE, F_E=F_E, F_k=F_k, a=a, x=pl['x'][:N_eff],
                    plasma=pl, info=info)
