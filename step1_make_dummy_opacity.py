"""
step1_make_dummy_opacity.py

Generate a FICTITIOUS 1500-group opacity / emissivity table.

    *** The numbers in this table have NO physical meaning. ***

The real tin table used in the paper cannot be distributed.  This script makes
a stand-in table for a fictitious element "X" so that the whole workflow
(fine-group transport -> generalized groups -> few-group table) can be run and
tested by anyone.  The table only imitates the qualitative features that make
the problem difficult:

  * a smooth continuum (free-free-like and bound-free-like),
  * several unresolved transition arrays (UTAs) for every charge state, as in
    tin or gold: arrays that stay in almost the same band for many charge
    states, and arrays that move quickly with the charge state,
  * a mean charge <Z>(T_e, n_i) that moves the line bundles with the plasma state,
  * a non-LTE-like emissivity (below the Planck limit at low density).

All line positions, widths and strengths are random numbers (fixed seed).

Output (NumPy .npz), default name  dummy_opacity_1500g.npz
    Z_nuc, A          fictitious atomic number and mass number
    E, dE, E_edges    photon-energy mesh [eV]  (G = 1500 groups, log-spaced)
    Te                electron temperature grid [eV]      (uniform in log)
    Ni                ion density grid [cm^-3]            (uniform in log)
    Zbar[iTe, iNi]    mean charge
    opac[iTe, iNi, iE]  absorption coefficient kappa [1/cm]
    emis[iTe, iNi, iE]  emissivity eps [erg/cm^3/s/eV]  (integrated over 4 pi)

Usage:
    python3 step1_make_dummy_opacity.py            # table + fig1_dummy_opacity.png
    python3 step1_make_dummy_opacity.py --seed 7   # another random table
"""
import argparse
import numpy as np

C_LIGHT = 2.99792458e10     # cm/s
EV_ERG = 1.602176634e-12    # erg
H_PLANCK = 6.62607015e-27   # erg s

Z_NUC = 40                  # fictitious element "X"
A_MASS = 100.0
Z_MAX = 30                  # highest charge state in the dummy model


def planck_4piB(E_eV, T_eV):
    """4 pi B_E [erg/cm^2/s/eV] (Planck function integrated over solid angle)."""
    x = np.minimum(E_eV / T_eV, 120.0)          # avoid underflow to exactly zero
    Ee = E_eV * EV_ERG
    return 4 * np.pi * 2 * Ee ** 3 / (H_PLANCK ** 3 * C_LIGHT ** 2) / np.expm1(x) * EV_ERG


def mean_charge(Te, Ni):
    """Smooth, monotonic dummy <Z>(T_e, n_i).  Te, Ni: 2-D arrays."""
    x = (Te / 35.0) ** 1.3 * (Ni / 1e20) ** (-0.06)
    return 1.0 + (Z_MAX - 1.0) * x / (1.0 + x)


# Unresolved transition arrays (UTAs) of the dummy ion, imitating the structure
# of mid- and high-Z plasmas such as tin and gold.  Each charge state q has
# every array below; one array is a dense bundle of random lines.
#   centre(q) = E0 + E1 q + E2 q^2 [eV],  rel_width = width of the bundle / centre
# "dn = 0" arrays move only a little with q, so the arrays of many charge
# states pile up in the same narrow band (like the 4p-4d / 4d-4f arrays of tin
# near 13.5 nm).  "dn = 1, 2" arrays move quickly with q.
#            name                E0     E1    E2   rel_width  strength  n_lines
UTA_ARRAYS = (('dn=0 (a)',       62.0,  1.3, 0.00,   0.035,     1.00,     120),
              ('dn=0 (b)',      105.0,  2.2, 0.00,   0.030,     0.50,     100),
              ('dn=1 (a)',       25.0,  9.0, 0.22,   0.045,     0.40,     120),
              ('dn=1 (b)',       45.0, 15.0, 0.35,   0.050,     0.15,     100),
              ('dn=2',          110.0, 30.0, 0.80,   0.060,     0.04,      80))


def line_cross_sections(E, rng):
    """Random UTAs for each charge state q = 0..Z_MAX.
    Returns S[q, iE] [cm^2]: absorption cross-section per ion of charge q."""
    S = np.zeros((Z_MAX + 1, E.size))
    for q in range(Z_MAX + 1):
        for _, E0, E1, E2, rel_width, amp, n_lines in UTA_ARRAYS:
            centre = E0 + E1 * q + E2 * q ** 2
            pos = centre * (1.0 + rel_width * rng.standard_normal(n_lines))    # line positions
            width = pos * rng.uniform(0.002, 0.006, n_lines)                   # Gaussian line widths
            strength = amp * 6e-17 * rng.lognormal(0.0, 1.3, n_lines)          # peak [cm^2]
            S[q] += (strength[:, None] * np.exp(-0.5 * ((E[None, :] - pos[:, None]) / width[:, None]) ** 2)).sum(axis=0)
    return S


def boundfree_cross_sections(E):
    """Dummy photo-ionization edges.  Returns S[q, iE] [cm^2]."""
    q = np.arange(Z_MAX + 1)[:, None]
    I_q = 8.0 + 4.0 * q + 0.3 * q ** 2                         # edge energy [eV]
    return 2e-19 * np.where(E[None, :] >= I_q, (E[None, :] / I_q) ** (-3.0), 0.0)


def make_table(seed=0, n_groups=1500, n_Te=44, n_Ni=81):
    rng = np.random.default_rng(seed)
    E_edges = np.geomspace(1.0, 2000.0, n_groups + 1)
    E = np.sqrt(E_edges[1:] * E_edges[:-1])
    dE = np.diff(E_edges)
    Te = np.geomspace(1.0, 300.0, n_Te)
    Ni = 10.0 ** (15.0 + 0.1 * np.arange(n_Ni))                # 1e15 ... 1e23
    TT, NN = np.meshgrid(Te, Ni, indexing='ij')
    Zbar = mean_charge(TT, NN)

    # charge-state distribution: Gaussian around <Z>
    q = np.arange(Z_MAX + 1)
    P = np.exp(-0.5 * ((q[None, None, :] - Zbar[:, :, None]) / 1.2) ** 2)
    P /= P.sum(axis=2, keepdims=True)                          # [iTe, iNi, q]

    sigma = line_cross_sections(E, rng) + boundfree_cross_sections(E)   # [q, iE]
    opac = NN[:, :, None] * (P @ sigma)                        # bound-bound + bound-free
    # free-free-like continuum
    Ne = Zbar * NN
    opac += (2.4e-37 * Zbar ** 2 * Ne * NN / np.sqrt(TT))[:, :, None] / E[None, None, :] ** 3
    # stimulated-emission correction
    opac *= -np.expm1(-E[None, None, :] / TT[:, :, None])

    # emissivity: Kirchhoff's law times a non-LTE-like reduction at low density
    nonlte = Ne / (Ne + 3e19)
    emis = opac * planck_4piB(E[None, None, :], TT[:, :, None]) * nonlte[:, :, None]
    return dict(Z_nuc=Z_NUC, A=A_MASS, E=E, dE=dE, E_edges=E_edges, Te=Te, Ni=Ni,
                Zbar=Zbar, opac=opac, emis=emis,
                label='DUMMY table for fictitious element X - no physical meaning (seed=%d)' % seed)


def plot_table(t, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    E, Te, Ni = t['E'], t['Te'], t['Ni']
    iN = np.argmin(np.abs(Ni - 1e20))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    for T in (10, 30, 100):
        iT = np.argmin(np.abs(Te - T))
        ax[0].loglog(E, t['opac'][iT, iN], lw=0.7, label='T$_e$ = %.0f eV' % Te[iT])
        ax[1].loglog(E, t['emis'][iT, iN], lw=0.7, label='T$_e$ = %.0f eV' % Te[iT])
    ax[0].set_ylabel('absorption coefficient $\\kappa$ [1/cm]')
    ax[1].set_ylabel('emissivity $\\varepsilon$ [erg/cm$^3$/s/eV]')
    top = t['emis'][:, iN].max()
    ax[1].set_ylim(top * 1e-12, top * 3)
    for a in ax[:2]:
        a.set_xlabel('photon energy [eV]')
        a.set_title('n$_i$ = %.0e cm$^{-3}$' % Ni[iN])
        a.legend()
        a.grid(alpha=0.3)
    pc = ax[2].pcolormesh(Ni, Te, t['Zbar'], shading='auto', cmap='viridis')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('ion density n$_i$ [cm$^{-3}$]')
    ax[2].set_ylabel('T$_e$ [eV]')
    ax[2].set_title('mean charge <Z>')
    fig.colorbar(pc, ax=ax[2])
    fig.suptitle('Dummy opacity table (fictitious element X, %d groups) - no physical meaning' % E.size)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=0, help='random seed of the dummy line data')
    p.add_argument('--groups', type=int, default=1500, help='number of fine photon-energy groups')
    p.add_argument('--out', default='dummy_opacity_1500g.npz')
    p.add_argument('--fig', default='fig1_dummy_opacity.png')
    a = p.parse_args()
    table = make_table(a.seed, a.groups)
    np.savez(a.out, **table)
    plot_table(table, a.fig)
    print('wrote %s  (opac/emis shape = %s)' % (a.out, table['opac'].shape))
    print('wrote %s' % a.fig)
