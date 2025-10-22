import numpy as np
import pyccl.nl_pt as pt
import scipy

def model_wgg_phot_lightcone(p, config_setup):
    
    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)
    
    # Calculate some power spectra with FAST-PT
    # Galaxies x galaxies.
    pk_gg = pt.get_pt_pk2d(config_setup['cosmo'], ptt_g, ptc=config_setup['ptc_gg'])
    # Galaxies x matter
    pk_gm = pt.get_pt_pk2d(config_setup['cosmo'], ptt_g, tracer2=config_setup['ptt_m'], ptc=config_setup['ptc_gg'])

    # I evaluate the power spectrum
    pk_gg_z = np.zeros_like(config_setup['k'])
    pk_gm_z = np.zeros_like(config_setup['k'])
    pk_mm_z = np.zeros_like(config_setup['k'])
    for i, z_i in enumerate(config_setup['z_centers']):
        pk_gg_z[i, :] = pk_gg.eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        pk_gm_z[i, :] = pk_gm.eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        pk_mm_z[i, :] = config_setup['pk_mm'].eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])

    def chunk_cl_integrals_wgg(zm_chunk_size, l_chunk_size):

        zm_chunks = [range(i, min(i + zm_chunk_size, len(config_setup['zm_centers']))) for i in range(0, len(config_setup['zm_centers']), zm_chunk_size)]
        l_chunks = [range(i, min(i + l_chunk_size, len(config_setup['l']))) for i in range(0, len(config_setup['l']), l_chunk_size)]
    
        cl_wgg = np.zeros((len(config_setup['Pi']), len(config_setup['zm_centers']), len(config_setup['l'])))
    
        for zm_chunk in zm_chunks:
            for l_chunk in l_chunks:
                cl_gg_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_wgg'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_gg_z[:, l_chunk])
                cl_mm_integrand = 4*(config_setup['alpha']-1)*(config_setup['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['lensing_z1_lensing_z2_wgg'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_mm_z[:, l_chunk])
                cl_gm_integrand = 2*(config_setup['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_z1_lensing_z2_wgg'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_gm_z[:, l_chunk])
                cl_mg_integrand = 2*(config_setup['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_z2_lensing_z1_wgg'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_gm_z[:, l_chunk])
                cl_wgg[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gg_integrand+cl_mm_integrand+cl_gm_integrand+cl_mg_integrand, config_setup['cmd'], axis = 0)
    
        return cl_wgg

    zm_chunk_size = 2
    l_chunk_size = 101
    cl_wgg = chunk_cl_integrals_wgg(zm_chunk_size, l_chunk_size)
    theta = np.einsum('i, j->ij', config_setup['rp_model'], 1/config_setup['y3fid'].comoving_distance(config_setup['zm_centers']).value)
    
    #Hankel transform
    integrand_wgg = np.einsum('ijk, ljk->iljk', config_setup['l']*scipy.special.j0(np.einsum('i, jk->jki', config_setup['l'], theta)), cl_wgg)/(2*np.pi)
    
    corr_function_phot_wgg = np.trapz(integrand_wgg, config_setup['l'], axis = 3)
    
    #Integration over zm
    zm_integration_phot_wgg = np.trapz(np.einsum('i,jki->jki', config_setup['kernel_wgg'], corr_function_phot_wgg), config_setup['zm_centers'], axis = 2)
    
    #Integration over pi
    wgg_phot = np.trapz(zm_integration_phot_wgg, config_setup['Pi'], axis = 1)
    
    return wgg_phot.value
    