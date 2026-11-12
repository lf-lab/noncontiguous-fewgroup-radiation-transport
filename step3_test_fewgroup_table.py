"""
step3_test_fewgroup_table.py

Test the K-group table against the fine-group calculation on cases that were
NOT used for training.

For each test case the outgoing spectrum at x = x_max is computed with
  (a) the fine-group solver                 (reference, 1500 groups),
  (b) the non-contiguous K-group table      (proposed),
  (c) the conventional K-group table        (contiguous bands, flat shape),
and the errors of (b) and (c) are evaluated:
  relL2 : relative L2 error of the spectrum on the fine mesh
  total : relative error of the spectrally integrated flux

Output
  test_results.csv          errors and wall time of every test case
  fig3_spectra.png          spectra of the best / median / worst test case
  fig4_errors.png           error distributions and computing time

Usage:
    python3 step3_test_fewgroup_table.py
    python3 step3_test_fewgroup_table.py --K 150 --n_test 40
    python3 step3_test_fewgroup_table.py --case 10.6 30 10      # one case: lambda_L T_e t_L
"""
import argparse
import csv
import time
import numpy as np

import mg_diffusion as M
import fewgroup as FG


def frange(a, b, step):
    return np.round(a + step * np.arange(int(round((b - a) / step)) + 1), 10)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--K', type=int, default=150)
    p.add_argument('--table', default='dummy_opacity_1500g.npz', help='fine-group table (reference)')
    p.add_argument('--fewgroup', default=None, help='default: fewgroup_table_K<K>.npz')
    p.add_argument('--conventional', default=None, help='default: conventional_table_K<K>.npz')
    p.add_argument('--n_test', type=int, default=20, help='number of random test cases')
    p.add_argument('--case', nargs=3, type=float, default=None, metavar=('LAMBDA_L', 'T_E', 'T_L'),
                   help='test only this case instead of random cases')
    p.add_argument('--limiter', default='larsen', choices=['none', 'larsen', 'levermore'])
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()

    tab = M.OpacityTable(a.table)
    fg = FG.FewGroupModel(a.fewgroup or 'fewgroup_table_K%d.npz' % a.K)
    cv = FG.FewGroupModel(a.conventional or 'conventional_table_K%d.npz' % a.K)
    z = np.load(a.fewgroup or 'fewgroup_table_K%d.npz' % a.K)
    n_i = float(z['n_i'])
    E, dE = tab.E, tab.dE

    # ---- test cases: grid points that were not used for training -----------
    if a.case is not None:
        test = [tuple(a.case)]
    else:
        Tes, lams, tLs = (frange(*g) for g in z['grid'])
        train = {tuple(c) for c in z['train_cases']}
        cases = [(float(l), float(T), float(t)) for T in Tes for l in lams for t in tLs]
        cases = [c for c in cases if c not in train]
        rng = np.random.default_rng(a.seed)
        test = [cases[i] for i in rng.permutation(len(cases))[:a.n_test]]
    print('%d test case(s); fine table: %d groups; K = %d' % (len(test), E.size, fg.K))

    rows, spectra = [], []
    for lam, Te, tL in test:
        t0 = time.time()
        ref = M.run(lam, n_i, Te, tL, tab, limiter=a.limiter, verbose=False)
        t_full = time.time() - t0
        o_fg = fg.run(lam, n_i, Te, tL, limiter=a.limiter)
        o_cv = cv.run(lam, n_i, Te, tL, limiter=a.limiter)
        e_fg = M.spectrum_errors(ref['F_E'], o_fg['F_E'], dE)
        e_cv = M.spectrum_errors(ref['F_E'], o_cv['F_E'], dE)
        rows.append(dict(lambda_L=lam, T_e=Te, t_L=tL,
                         relL2_noncontiguous=e_fg[0], total_noncontiguous=e_fg[1],
                         relL2_conventional=e_cv[0], total_conventional=e_cv[1],
                         t_fine=t_full, t_noncontiguous=o_fg['info']['time'],
                         t_conventional=o_cv['info']['time']))
        spectra.append((ref['F_E'], o_fg['F_E'], o_cv['F_E']))
        print('lambda_L = %5.2f um, T_e = %5.1f eV, t_L = %5.1f ns | non-contiguous: relL2 %.3f, total %.4f'
              ' | conventional: relL2 %.3f, total %.4f' % (lam, Te, tL, e_fg[0], e_fg[1], e_cv[0], e_cv[1]))

    with open('test_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows])
    print('\n%-16s %10s %10s %10s %10s %12s' % ('', 'relL2 med', 'relL2 p90', 'total med', 'total p90', 'time/case'))
    for name in ('noncontiguous', 'conventional'):
        print('%-16s %10.3f %10.3f %10.4f %10.4f %10.4f s'
              % (name, np.median(col('relL2_' + name)), np.percentile(col('relL2_' + name), 90),
                 np.median(col('total_' + name)), np.percentile(col('total_' + name), 90),
                 np.median(col('t_' + name))))
    print('%-16s %54.4f s' % ('fine (%d groups)' % E.size, np.median(col('t_fine'))))
    print('speed-up of the K-group table: x %.0f' % (np.median(col('t_fine')) / np.median(col('t_noncontiguous'))))

    # ---- figures ------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order = np.argsort(col('relL2_noncontiguous'))
    pick = sorted({int(order[0]), int(order[len(order) // 2]), int(order[-1])},
                  key=lambda i: rows[i]['relL2_noncontiguous'])
    fig, axs = plt.subplots(len(pick), 2, figsize=(14, 3.8 * len(pick)), squeeze=False)
    for ax_row, i in zip(axs, pick):
        F_ref, F_fg, F_cv = spectra[i]
        r = rows[i]
        peak = F_ref.max() * 1e-7
        in_peak = E[F_ref > 0.02 * F_ref.max()]                # zoom range around the main emission
        for j, ax in enumerate(ax_row):
            ax.plot(E, F_ref * 1e-7, 'k', lw=0.9, label='fine (%d groups)' % E.size)
            ax.plot(E, F_cv * 1e-7, color='tab:blue', lw=0.8, label='conventional (K = %d)' % cv.K)
            ax.plot(E, F_fg * 1e-7, color='tab:red', lw=0.8, label='non-contiguous (K = %d)' % fg.K)
            ax.set_xlabel('photon energy [eV]')
            ax.set_ylabel('F$_E$ [W/cm$^2$/eV]')
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_ylim(peak * 1e-6, peak * 3)
                ax.set_title('$\\lambda_L$ = %g um, T$_e$ = %g eV, t$_L$ = %g ns' % (r['lambda_L'], r['T_e'], r['t_L']))
                ax.legend(fontsize=8)
            else:
                ax.set_xlim(in_peak.min(), in_peak.max())
                ax.set_title('relL2: non-contiguous %.3f, conventional %.3f'
                             % (r['relL2_noncontiguous'], r['relL2_conventional']))
    fig.tight_layout()
    fig.savefig('fig3_spectra.png', dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    names = ['non-contiguous', 'conventional']
    for a_, key, title in ((ax[0], 'relL2_', 'relative L2 error of the spectrum'),
                           (ax[1], 'total_', 'relative error of the total flux')):
        data = [np.maximum(col(key + 'noncontiguous'), 1e-6), np.maximum(col(key + 'conventional'), 1e-6)]
        a_.boxplot(data, whis=(0, 100))
        a_.set_xticks([1, 2])
        a_.set_xticklabels(names)
        for k, d in enumerate(data):
            a_.plot(np.full(d.size, k + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, d.size), d, 'o',
                    ms=3, color=('tab:red', 'tab:blue')[k], alpha=0.6)
        a_.set_yscale('log')
        a_.set_title(title)
        a_.grid(alpha=0.3, which='both')
    t = [np.median(col('t_fine')), np.median(col('t_conventional')), np.median(col('t_noncontiguous'))]
    ax[2].bar(['fine\n(%d groups)' % E.size, 'conventional\n(K = %d)' % cv.K, 'non-contiguous\n(K = %d)' % fg.K],
              t, color=['k', 'tab:blue', 'tab:red'])
    ax[2].set_yscale('log')
    ax[2].set_ylabel('wall time per case [s]')
    ax[2].set_title('computing time (median of %d cases)' % len(rows))
    ax[2].grid(alpha=0.3, which='both', axis='y')
    fig.tight_layout()
    fig.savefig('fig4_errors.png', dpi=130)
    print('wrote test_results.csv, fig3_spectra.png, fig4_errors.png')


if __name__ == '__main__':
    main()
