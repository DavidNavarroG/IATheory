import numpy as np
import pyccl.nl_pt as pt
import scipy


def model_wgp_spec_snapshot(p, config_setup):

    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Biases for IAs.
    if config_setup['IA_model'] == 'NLA':
        a_1 = p[2]
        a_2 = 0.0
        a_d = 0.0
    else:   
        a_1 = p[2]
        a_2 = p[3]
        a_d = p[4]

    z_i = config_setup['z_snapshot']
    # Let's convert the a_IA values into the correctly normalized c_IA values

    c_1,c_d,c_2 = pt.translate_IA_norm(config_setup['cosmo'], z=z_i, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)
    #ptt_g = pt.PTNumberCountsTracer(b1=b_1)
    
    # Intrinsic alignments
    if config_setup['IA_model'] == 'NLA':
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=c_1)
    else:
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=c_1, c2=c_2, cdelta=c_d)
    
    pk_gi = config_setup['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=ptt_i)
    
    pk_gi_z = pk_gi(config_setup['k_model'],1./(1+z_i), config_setup['cosmo'])
    integrand = pk_gi_z*config_setup['k_model']*scipy.special.jv(2, np.array([config_setup['k_model']*rp_i for rp_i in config_setup['rp_model']]))/(2*np.pi)
    wgp_spec_snapshot = -np.trapz(integrand, config_setup['k_model'], axis = 1)

    return wgp_spec_snapshot