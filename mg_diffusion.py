"""
mg_diffusion.py

Fine-group (reference) solver: steady, one-dimensional (slab), multigroup
flux-limited diffusion of radiation in an isothermally expanding plasma.

Input
    lambda_L [um]    laser wavelength -> critical density n_c = 1.1e21 / lambda_L^2 [cm^-3]
    n_i      [cm^-3] ion density before expansion (solid density)
    T_e      [eV]    electron temperature (uniform)
    t_L      [ns]    laser pulse duration = expansion time
    table            fine-group opacity table (.npz made by step1_make_dummy_opacity.py)

Model
    Isothermal expansion:  n_i(x) = n_ic exp(-x / L),  L = c_s t_L,
                           c_s = sqrt(<Z> T_e / m_i)
    x = 0 is the critical surface: <Z>(T_e, n_ic) n_ic = n_c.
    Transport for each group g (E_g: radiation energy density of the group):
        -d/dx[ D_g dE_g/dx ] + c kappa_g E_g = eps_g
        D_g = c lambda(R) / kappa_g,   R = |dE_g/dx| / (kappa_g E_g)
    Flux limiter: Larsen (n = 2, default), Levermore-Pomraning, or none.
    Boundaries: Marshak vacuum boundary (no incoming radiation) on both sides
                (the x = 0 side can also be reflective).
    The region with n_i below the lowest table density is treated as vacuum
    and removed from the calculation (the flux is conserved there in a slab).

Numerics
    Cell-centred finite volume, Thomas algorithm vectorized over the groups,
    Picard iteration for the flux limiter.

Output
    Spectrum of the outgoing flux at x = x_max, F_E [erg/cm^2/s/eV].

Usage (single case):
    python3 mg_diffusion.py --lambda_L 10.6 --T_e 30 --t_L 10
"""
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import brentq

C_LIGHT = 2.99792458e10     # cm/s
EV_ERG = 1.602176634e-12    # erg
AMU_G = 1.66053907e-24      # g


# --------------------------------------------------------------------------
# Fine-group table and its interpolation
# --------------------------------------------------------------------------
class OpacityTable:
    """Fine-group table with cubic-spline interpolation in (log10 Te, log10 ni)."""

    def __init__(self, path, kappa_floor=1e-20, eps_floor=1e-40):
        t = np.load(path, allow_pickle=False)
        self.E, self.dE, self.E_edges = t['E'], t['dE'], t['E_edges']
        self.Te, self.Ni = t['Te'], t['Ni']
        self.A = float(t['A'])
        self.Zbar = t['Zbar']
        self.lTe, self.lNi = np.log10(self.Te), np.log10(self.Ni)
        self._zbar = RectBivariateSpline(self.lTe, self.lNi, self.Zbar, kx=3, ky=3)
        # kappa and eps are interpolated as log10 values, shape [iTe, iNi, iE]
        self.lkap = np.log10(np.maximum(t['opac'], kappa_floor))
        self.leps = np.log10(np.maximum(t['emis'], eps_floor))
        # clip range of the interpolated values (per group): prevents spline overshoot
        self.lkap_rng = (self.lkap.min(axis=(0, 1)), self.lkap.max(axis=(0, 1)))
        self.leps_rng = (self.leps.min(axis=(0, 1)), self.leps.max(axis=(0, 1)))
        self._cache_Te = None

    def zbar(self, Te, ni):
        lTe = np.clip(np.log10(Te), self.lTe[0], self.lTe[-1])
        lni = np.clip(np.log10(ni), self.lNi[0], self.lNi[-1])
        return self._zbar(lTe, lni, grid=False)

    def _fix_Te(self, Te):
        """Evaluate the spline in the Te direction once, keep splines in ni."""
        if self._cache_Te == Te:
            return
        lTe = float(np.clip(np.log10(Te), self.lTe[0], self.lTe[-1]))
        kap_Te = CubicSpline(self.lTe, self.lkap, axis=0)(lTe)   # [iNi, iE]
        eps_Te = CubicSpline(self.lTe, self.leps, axis=0)(lTe)
        self._kap_ni = CubicSpline(self.lNi, kap_Te, axis=0)
        self._eps_ni = CubicSpline(self.lNi, eps_Te, axis=0)
        self._cache_Te = Te

    def kappa_eps(self, Te, ni):
        """Return kappa [1/cm] and eps [erg/cm^3/s/eV], shape (len(ni), G).
        Below the lowest table density: kappa ~ ni, eps ~ ni^2."""
        self._fix_Te(Te)
        ni = np.atleast_1d(np.asarray(ni, float))
        lni = np.log10(ni)
        lni_c = np.clip(lni, self.lNi[0], self.lNi[-1])
        kap = 10 ** np.clip(self._kap_ni(lni_c), *self.lkap_rng)
        eps = 10 ** np.clip(self._eps_ni(lni_c), *self.leps_rng)
        scale = 10 ** (lni - lni_c)
        kap *= scale[:, None]
        eps *= (scale ** 2)[:, None]
        return kap, eps


# --------------------------------------------------------------------------
# Plasma profile
# --------------------------------------------------------------------------
def critical_density(lambda_L_um):
    return 1.1e21 / lambda_L_um ** 2


def build_plasma(tab, lambda_L, n_i, T_e, t_L, x_max=500.0, delta_x=1.0, verbose=True):
    """Isothermal expansion profile.  `tab` needs .zbar(Te, ni), .A and .Ni
    (both the fine table and the few-group table provide them)."""
    n_c = critical_density(lambda_L)
    m_i = tab.A * AMU_G
    # ion density at the critical surface: <Z>(Te, n_ic) n_ic = n_c
    g = lambda lni: float(tab.zbar(T_e, 10 ** lni)) * 10 ** lni - n_c
    if g(np.log10(n_i)) < 0:
        raise ValueError('n_e at the input density is below the critical density')
    n_ic = 10 ** brentq(g, np.log10(tab.Ni[0]), np.log10(n_i))
    Z_c = float(tab.zbar(T_e, n_ic))
    c_s = np.sqrt(Z_c * T_e * EV_ERG / m_i)           # sound speed [cm/s]
    L = c_s * t_L * 1e-9                              # density scale length [cm]
    N = int(round(x_max / delta_x))
    x = (np.arange(N) + 0.5) * delta_x                # cell centres [um]
    ni_x = n_ic * np.exp(-x * 1e-4 / L)
    Z_x = tab.zbar(T_e, ni_x)
    if verbose:
        print('n_c = %.3e cm^-3 (lambda_L = %g um)' % (n_c, lambda_L))
        print('critical surface: n_i = %.3e cm^-3, <Z> = %.2f' % (n_ic, Z_c))
        print('c_s = %.3e cm/s,  L = c_s t_L = %.1f um' % (c_s, L * 1e4))
    return dict(x=x, dx=delta_x * 1e-4, ni=ni_x, ne=Z_x * ni_x, Z=Z_x, n_c=n_c, n_ic=n_ic,
                Z_c=Z_c, c_s=c_s, L=L, lambda_L=lambda_L, T_e=T_e, t_L=t_L)


# --------------------------------------------------------------------------
# Multigroup flux-limited diffusion solver
# --------------------------------------------------------------------------
def solve_tridiag_vec(a, b, c, d):
    """Thomas algorithm vectorized over the groups. a, c: (N-1, G); b, d: (N, G)."""
    n = b.shape[0]
    cp = np.empty_like(a)
    dp = np.empty_like(d)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i - 1] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    x = np.empty_like(d)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def flux_limited_D(kap, R, limiter='larsen', n_larsen=2):
    """Flux-limited diffusion coefficient D = c lambda(R) / kappa.
      'none'      : lambda = 1/3
      'larsen'    : lambda = (3^n + R^n)^(-1/n)
      'levermore' : lambda = (coth R - 1/R) / R      (Levermore-Pomraning)
    All give lambda -> 1/3 for R -> 0 and lambda -> 1/R for R -> infinity."""
    if limiter == 'none':
        return C_LIGHT / (3.0 * kap)
    if limiter == 'larsen':
        if n_larsen == 2:
            return C_LIGHT / (kap * np.sqrt(9.0 + R * R))
        return C_LIGHT / (kap * (3.0 ** n_larsen + R ** n_larsen) ** (1.0 / n_larsen))
    if limiter == 'levermore':
        Rs = np.maximum(R, 1e-6)
        lam = np.where(R < 1e-3, 1.0 / 3.0 - Rs ** 2 / 45.0, (1.0 / np.tanh(Rs) - 1.0 / Rs) / Rs)
        return C_LIGHT * lam / kap
    raise ValueError(limiter)


def solve_diffusion(kap_abs, kap_dif, src, dx, left_bc='vacuum', limiter='larsen', n_larsen=2,
                    max_iter=200, tol=1e-5, relax=1.0):
    """Solve  -d/dx[D du/dx] + c kap_abs u = src  for many independent groups.

    The same routine is used for the fine groups (kap_abs = kap_dif = kappa_g)
    and for the generalized groups (kap_abs = kappa_bar_k, kap_dif = kappa_D,k).

    kap_abs, kap_dif, src : (N, G)       dx : [cm]
    Returns u (N, G), F_right (G,) outgoing flux at the right boundary, info dict.
    Marshak boundary (no incoming flux): F_out = (c/2) u_b,
        u_b = u_N / (1 + c dx / (4 D_N)).
    """
    N, G = kap_abs.shape
    kD_f = 0.5 * (kap_dif[1:] + kap_dif[:-1])                 # kappa on the cell faces
    ckap = C_LIGHT * kap_abs
    R_f = np.zeros((N - 1, G))
    R_L = np.zeros(G)
    R_R = np.zeros(G)
    Df = None
    u_old = None
    F_old = None
    err = np.nan
    err_prev = np.inf
    for it in range(max_iter):
        Df_n = flux_limited_D(kD_f, R_f, limiter, n_larsen)
        D_R_n = flux_limited_D(kap_dif[-1], R_R, limiter, n_larsen)
        D_L_n = flux_limited_D(kap_dif[0], R_L, limiter, n_larsen)
        if Df is None or relax == 1.0:
            Df, D_R, D_L = Df_n, D_R_n, D_L_n
        else:                                                 # under-relaxation in log space
            Df = Df ** (1 - relax) * Df_n ** relax
            D_R = D_R ** (1 - relax) * D_R_n ** relax
            D_L = D_L ** (1 - relax) * D_L_n ** relax
        lo = Df * (-1.0 / dx ** 2)
        b = ckap.copy()
        b[1:] -= lo
        b[:-1] -= lo
        gR = (C_LIGHT / 2) / (1 + C_LIGHT * dx / (4 * D_R)) / dx
        b[-1] += gR
        if left_bc == 'vacuum':
            b[0] += (C_LIGHT / 2) / (1 + C_LIGHT * dx / (4 * D_L)) / dx
        elif left_bc != 'reflective':
            raise ValueError(left_bc)
        u = solve_tridiag_vec(lo, b, lo, src)
        np.maximum(u, 0.0, out=u)
        if limiter == 'none':
            break
        F_new = gR * dx * u[-1]
        if u_old is not None:
            # convergence: change of u weighted by the absorption rate, and change of the boundary flux
            err_u = np.max((kap_abs * np.abs(u - u_old)).max(axis=0) / np.maximum((kap_abs * u).max(axis=0), 1e-300))
            err_F = np.max(np.abs(F_new - F_old) / np.maximum(F_new, 1e-300))
            err = max(err_u, err_F)
            if err < tol:
                break
            if err > err_prev:                                # oscillation -> stronger relaxation
                relax = max(relax * 0.8, 0.2)
            err_prev = err
        u_old = u
        F_old = F_new
        # new R = |du/dx| / (kappa u) on the faces
        u_f = 0.5 * (u[1:] + u[:-1])
        du = np.abs(u[1:] - u[:-1])
        du = du * du ** 2 / (du ** 2 + (1e-8 * u_f) ** 2)     # suppress round-off level gradients
        R_f = du / dx / np.maximum(kD_f * u_f, 1e-300)
        R_R = R_f[-1]                                         # boundary R from the last face
        R_L = R_f[0]
    F_right = gR * dx * u[-1]
    info = dict(n_iter=it + 1, err=err, converged=bool(err < tol) or limiter == 'none')
    return u, F_right, info


def run(lambda_L, n_i, T_e, t_L, table, delta_x=1.0, x_max=500.0, left_bc='vacuum',
        limiter='larsen', n_larsen=2, kappa_min=1e-5, verbose=True):
    """Fine-group calculation of one case.  `table`: path or OpacityTable.
    kappa_min: floor of kappa [1/cm] (smaller values make the matrix ill-conditioned)."""
    tab = table if isinstance(table, OpacityTable) else OpacityTable(table)
    pl = build_plasma(tab, lambda_L, n_i, T_e, t_L, x_max, delta_x, verbose)
    N_eff = int(np.sum(pl['ni'] >= tab.Ni[0]))                # cells inside the table range
    kap, eps = tab.kappa_eps(T_e, pl['ni'][:N_eff])           # (N_eff, G)
    kap = np.maximum(kap, kappa_min)
    E_rad, F_right, info = solve_diffusion(kap, kap, eps * tab.dE[None, :], pl['dx'],
                                           left_bc, limiter, n_larsen)
    F_E = F_right / tab.dE                                    # erg/cm^2/s/eV
    if verbose:
        print('outgoing flux at x_max = %.3e W/cm^2 (%d Picard iterations)'
              % (F_right.sum() * 1e-7, info['n_iter']))
    return dict(E_eV=tab.E, dE=tab.dE, F_E=F_E, E_rad=E_rad, kappa=kap, eps=eps,
                x=pl['x'][:N_eff], plasma=pl, info=info)


def spectrum_errors(F_ref, F_red, dE):
    """(relative L2 error of the spectrum, relative error of the total flux)."""
    norm = np.sqrt((F_ref ** 2 * dE).sum())
    rel_l2 = np.sqrt(((F_ref - F_red) ** 2 * dE).sum()) / max(norm, 1e-300)
    total = abs(((F_red - F_ref) * dE).sum()) / max(abs((F_ref * dE).sum()), 1e-300)
    return rel_l2, total


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lambda_L', type=float, default=1.064, help='laser wavelength [um]')
    p.add_argument('--n_i', type=float, default=4e22, help='ion density before expansion [cm^-3]')
    p.add_argument('--T_e', type=float, default=30.0, help='electron temperature [eV]')
    p.add_argument('--t_L', type=float, default=10.0, help='pulse duration = expansion time [ns]')
    p.add_argument('--delta_x', type=float, default=1.0, help='cell size [um]')
    p.add_argument('--x_max', type=float, default=500.0, help='size of the domain [um]')
    p.add_argument('--left_bc', default='vacuum', choices=['vacuum', 'reflective'])
    p.add_argument('--limiter', default='larsen', choices=['none', 'larsen', 'levermore'])
    p.add_argument('--table', default='dummy_opacity_1500g.npz')
    p.add_argument('--prefix', default='spectrum_fine')
    a = p.parse_args()
    out = run(a.lambda_L, a.n_i, a.T_e, a.t_L, a.table, a.delta_x, a.x_max, a.left_bc, a.limiter)
    np.savetxt(a.prefix + '.txt', np.c_[out['E_eV'], out['F_E'], out['F_E'] * 1e-7],
               header='E[eV]  F_E[erg/cm2/s/eV]  F_E[W/cm2/eV]', fmt='%.6e')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(out['E_eV'], out['F_E'] * 1e-7, lw=0.8)
    ax.set_ylim(out['F_E'].max() * 1e-7 * 1e-6, out['F_E'].max() * 1e-7 * 3)
    ax.set_xlabel('photon energy [eV]')
    ax.set_ylabel('F$_E$ [W/cm$^2$/eV]')
    ax.set_title('lambda_L = %g um, T_e = %g eV, t_L = %g ns' % (a.lambda_L, a.T_e, a.t_L))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.prefix + '.png', dpi=130)
    print('wrote %s.txt and %s.png' % (a.prefix, a.prefix))
