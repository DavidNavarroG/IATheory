import numpy as np
import pyccl.nl_pt as pt
import scipy

def model_wgp_spec_lightcone(p, config_setup):

    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Biases for IAs.
    if config_setup['IA_model'] == 'NLA':
        a_1 = p[2]
        a_2 = 0
        a_d = 0
    else:
        a_1 = p[2]
        a_2 = p[3]
        a_d = p[4]
    
    # Let's convert the a_IA values into the correctly normalized c_IA values:
    c_1,c_d,c_2 = pt.translate_IA_norm(config_setup['cosmo'], z=config_setup['z_centers'], a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)
    
    # Intrinsic alignments
    if config_setup['IA_model'] == 'NLA':
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config_setup['z_centers'],c_1)) # NLA model
    else:
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config_setup['z_centers'],c_1), c2=(config_setup['z_centers'],c_2), cdelta=(config_setup['z_centers'],c_d)) # TATT model

    # Calculate some power spectra with FAST-PT
    # Galaxies x galaxies.
    pk_gi = config_setup['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=ptt_i)

    pk_gi_z = np.zeros_like(config_setup['k'])
    corr_function_spec = np.zeros((len(config_setup['rp_model']), len(config_setup['z_centers'])))
    for i, z_i in enumerate(config_setup['z_centers']):
        pk_gi_z[i, :] = pk_gi(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        cl = pk_gi_z[i, :]/(config_setup['y3fid'].comoving_distance(z_i).value**2)
        theta = config_setup['rp_model'] / config_setup['y3fid'].comoving_distance(z_i).value
        #Hankel transform
        integrand = (config_setup['l']*scipy.special.jv(2, np.array([config_setup['l'] * theta_i for theta_i in theta]))*cl)/(2*np.pi)
        corr_function_spec[:, i] = np.trapz(integrand, config_setup['l'], axis = 1)

    #Integration over z
    wgp_spec = -np.trapz(np.einsum('i,ji->ji', config_setup['kernel_wgp'], corr_function_spec), config_setup['z_centers'], axis = 1)

    return wgp_spec
