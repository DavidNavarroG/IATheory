import numpy as np
import pyccl.nl_pt as pt
import scipy


def model_wgg_spec_snapshot(p, config_setup):
    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)

    # Calculate some power spectra with FAST-PT
    # Galaxies x galaxies.
    pk_gg = pt.get_pt_pk2d(config_setup['cosmo'], ptt_g, ptc=config_setup['ptc_gg'], nonlin_pk_type='nonlinear')

    z_i = config_setup['z_snapshot']
    pk_gg_z = pk_gg.eval(config_setup['k_model'],1./(1+z_i), config_setup['cosmo'])
    integrand = pk_gg_z*config_setup['k_model']*scipy.special.j0(np.array([config_setup['k_model']*rp_i for rp_i in config_setup['rp_model']]))/(2*np.pi)
    wgg_spec_snapshot = np.trapz(integrand, config_setup['k_model'], axis = 1)

    return wgg_spec_snapshot
