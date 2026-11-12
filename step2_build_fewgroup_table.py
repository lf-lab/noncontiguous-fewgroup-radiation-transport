"""
step2_build_fewgroup_table.py

Build the K-group table (default K = 150) from the fine-group table.

Steps
  1. Choose a small random fraction (default 5 %) of the cases on the
     (T_e, lambda_L, t_L) grid and solve them with the fine-group solver
     (mg_diffusion.run).  The shape-normalized spectra u_g / integral(u dE)
     inside the plasma are accumulated into a Gram matrix.
  2. Find K non-contiguous generalized groups S_k and their shapes w_k(g)
     (fewgroup.noncontiguous_partition).
  3. Project the fine table onto the modes and save the K-group table.
     A conventional K-group table (contiguous bands, flat shape) is also
     saved for comparison.

Output
  fewgroup_table_K<K>.npz       proposed non-contiguous K-group table
  conventional_table_K<K>.npz   conventional K-group table (reference)
  fig2_generalized_groups.png   which fine group belongs to which generalized group

  Contents of the .npz tables:
    K, E, dE (fine mesh), label (G,), modes (K, G) = w_k, lTe, lNi (log10 grids),
    lkab, leps, lkD (nTe, nNi, K) = log10 of kappa_bar, eps_bar, kappa_D,
    Zbar (nTe, nNi), A, explained, train_cases, grid, n_i, seed

Usage:
    python3 step2_build_fewgroup_table.py
    python3 step2_build_fewgroup_table.py --K 100 --Te 5 250 5 --lambda_L 1 11 1 --t_L 1 100 5
"""
import argparse
import sys
import time
import numpy as np

import mg_diffusion as M
import fewgroup as FG


def frange(a, b, step):
    return np.round(a + step * np.arange(int(round((b - a) / step)) + 1), 10)


def plot_partition(E, dE, lab, modes, mean_shape, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    K = modes.shape[0]
    # sort the generalized groups by their mean photon energy
    mean_E = np.array([E[lab == k].mean() if np.any(lab == k) else 0.0 for k in range(K)])
    rank = np.empty(K, int)
    rank[np.argsort(mean_E)] = np.arange(K)
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax[0].scatter(E, mean_shape, c=rank[lab], cmap='turbo', s=5)
    ax[0].set_yscale('log')
    ax[0].set_ylim(mean_shape.max() * 1e-6, mean_shape.max() * 3)
    ax[0].set_ylabel('mean training spectrum shape [1/eV]')
    ax[0].set_title('colour = generalized group')
    ax[1].scatter(E, rank[lab], c=rank[lab], cmap='turbo', s=5)
    ax[1].set_ylabel('generalized group index (sorted by mean energy)')
    ax[1].set_xlabel('photon energy [eV]')
    ax[1].set_title('non-contiguous partition of %d fine groups into K = %d groups' % (E.size, K))
    ax[1].set_xscale('log')
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--K', type=int, default=150, help='number of generalized groups')
    p.add_argument('--table', default='dummy_opacity_1500g.npz', help='fine-group table')
    p.add_argument('--Te', nargs=3, type=float, default=[10, 100, 5], metavar=('MIN', 'MAX', 'STEP'),
                   help='electron temperature grid [eV]')
    p.add_argument('--lambda_L', nargs=3, type=float, default=[1, 11, 2], metavar=('MIN', 'MAX', 'STEP'),
                   help='laser wavelength grid [um]')
    p.add_argument('--t_L', nargs=3, type=float, default=[5, 50, 5], metavar=('MIN', 'MAX', 'STEP'),
                   help='pulse duration grid [ns]')
    p.add_argument('--train_frac', type=float, default=0.05, help='fraction of the grid used for training')
    p.add_argument('--x_stride', type=int, default=5, help='take a training sample every x_stride cells')
    p.add_argument('--n_i', type=float, default=4e22, help='ion density before expansion [cm^-3]')
    p.add_argument('--delta_x', type=float, default=1.0)
    p.add_argument('--x_max', type=float, default=500.0)
    p.add_argument('--limiter', default='larsen', choices=['none', 'larsen', 'levermore'])
    p.add_argument('--kappa_min', type=float, default=1e-5, help='floor of kappa [1/cm]')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--fig', default='fig2_generalized_groups.png')
    a = p.parse_args()

    tab = M.OpacityTable(a.table)
    E, dE = tab.E, tab.dE
    G = E.size

    # ---- 1. training cases ------------------------------------------------
    Tes, lams, tLs = frange(*a.Te), frange(*a.lambda_L), frange(*a.t_L)
    cases = [(float(l), float(T), float(t)) for T in Tes for l in lams for t in tLs]
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(cases))
    n_train = max(int(np.floor(a.train_frac * len(cases))), 1)
    train = sorted([cases[i] for i in perm[:n_train]], key=lambda c: (c[1], c[0], c[2]))
    print('grid %d x %d x %d = %d cases; training cases: %d (%.1f %%)'
          % (len(Tes), len(lams), len(tLs), len(cases), n_train, 100.0 * n_train / len(cases)))

    t0 = time.time()
    gram = np.zeros((G, G))
    mean_shape = np.zeros(G)
    n_samples = 0
    for i, (lam, Te, tL) in enumerate(train):
        out = M.run(lam, a.n_i, Te, tL, tab, a.delta_x, a.x_max, 'vacuum', a.limiter,
                    kappa_min=a.kappa_min, verbose=False)
        u = out['E_rad'][::a.x_stride] / dE[None, :]           # spectral energy density
        tot = u @ dE
        u = u[tot > 0] / tot[tot > 0, None]                    # normalize the shape
        gram += u.T @ u
        mean_shape += u.sum(axis=0)
        n_samples += u.shape[0]
        if (i + 1) % 10 == 0 or i + 1 == n_train:
            print('  training %d/%d (%.0f s)' % (i + 1, n_train, time.time() - t0))
            sys.stdout.flush()
    A = FG.gram_to_A(gram, n_samples, dE)

    # ---- 2. generalized groups --------------------------------------------
    t0 = time.time()
    lab, modes, frac = FG.noncontiguous_partition(A, dE, a.K, np.random.default_rng(a.seed))
    sizes = np.bincount(lab, minlength=a.K)
    print('K = %d generalized groups: explained energy %.5f, fine groups per group '
          'min/median/max = %d/%d/%d (%.1f s)'
          % (a.K, frac, sizes.min(), int(np.median(sizes)), sizes.max(), time.time() - t0))

    # ---- 3. tables --------------------------------------------------------
    extra = dict(train_cases=np.array(train), grid=np.array([a.Te, a.lambda_L, a.t_L]),
                 n_i=a.n_i, seed=a.seed)
    name = 'fewgroup_table_K%d.npz' % a.K
    FG.save_table(name, tab, lab, modes, frac, a.kappa_min, **extra)
    print('wrote', name)

    lab_c, modes_c, frac_c = FG.conventional_partition(A, dE, a.K)
    name_c = 'conventional_table_K%d.npz' % a.K
    FG.save_table(name_c, tab, lab_c, modes_c, frac_c, a.kappa_min, **extra)
    print('wrote %s (contiguous bands + flat shape: explained energy %.5f)' % (name_c, frac_c))

    plot_partition(E, dE, lab, modes, mean_shape / n_samples, a.fig)
    print('wrote', a.fig)


if __name__ == '__main__':
    main()
