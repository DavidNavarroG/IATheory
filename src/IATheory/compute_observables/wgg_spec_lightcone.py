import numpy as np
import pyccl.nl_pt as pt
import scipy

def model_wgg_spec_lightcone(p, config_setup):

    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)

    # Calculate some power spectra with FAST-PT
    # Galaxies x galaxies.
    pk_gg = config_setup['ptc_gg'].get_biased_pk2d(ptt_g)

    pk_gg_z = np.zeros_like(config_setup['k'])
    corr_function_spec = np.zeros((len(config_setup['rp_model']), len(config_setup['z_centers'])))
    for i, z_i in enumerate(config_setup['z_centers']):
        pk_gg_z[i, :] = pk_gg(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        cl = pk_gg_z[i, :]/(config_setup['y3fid'].comoving_distance(z_i).value**2)
        theta = config_setup['rp_model'] / config_setup['y3fid'].comoving_distance(z_i).value
        #Hankel transform
        integrand = (config_setup['l']*scipy.special.j0(np.array([config_setup['l'] * theta_i for theta_i in theta]))*cl)/(2*np.pi)
        corr_function_spec[:, i] = np.trapz(integrand, config_setup['l'], axis = 1)

    #Integration over z
    wgg_spec = np.trapz(np.einsum('i,ji->ji', config_setup['kernel_wgg'], corr_function_spec), config_setup['z_centers'], axis = 1)

    return wgg_spec
