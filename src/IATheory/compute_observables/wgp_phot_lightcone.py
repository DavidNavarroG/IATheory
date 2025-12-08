import numpy as np
import pyccl.nl_pt as pt
import scipy

def model_wgp_phot_lightcone(p, config_setup):
    
    # Galaxy biases   
    b_1 = p[0]
    b_2 = p[1]
    b_s = (-4/7)*(b_1 - 1)
    b_3nl = b_1 - 1

    # Number counts (galaxy clustering)
    ptt_g = pt.PTNumberCountsTracer(b1=b_1, b2=b_2, bs=b_s, b3nl = b_3nl)
    
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
    c_1,c_d,c_2 = pt.translate_IA_norm(config_setup['cosmo'], config_setup['z_centers'], a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)
    
    # Intrinsic alignments
    if config_setup['IA_model'] == 'NLA':
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config_setup['z_centers'],c_1)) # NLA model
    else:
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config_setup['z_centers'],c_1), c2=(config_setup['z_centers'],c_2), cdelta=(config_setup['z_centers'],c_d)) # TATT model
    
    # Calculate some power spectra with FAST-PT
    # Galaxies x intrinsic.
    pk_gi = pt.get_pt_pk2d(config_setup['cosmo'], ptt_g, tracer2=ptt_i, ptc=config_setup['ptc_gp'])
    pk_gi_z = np.zeros_like(config_setup['k'])

    if config_setup['add_galaxy_galaxy_lensing']:
        # Galaxies x matter
        pk_gm = pt.get_pt_pk2d(config_setup['cosmo'], ptt_g, tracer2=config_setup['ptt_m'], ptc=config_setup['ptc_gp'])
        pk_gm_z = np.zeros_like(config_setup['k'])

    if config_setup['add_magnification']:
        # Matter x intrinsic
        pk_mi = pt.get_pt_pk2d(config_setup['cosmo'], config_setup['ptt_m'], tracer2=ptt_i, ptc=config_setup['ptc_gp'])
        pk_mi_z = np.zeros_like(config_setup['k'])

    if config_setup['add_galaxy_galaxy_lensing'] and config_setup['add_magnification']:
        pk_mm_z = np.zeros_like(config_setup['k'])
        
    for i, z_i in enumerate(config_setup['z_centers']):
        pk_gi_z[i, :] = pk_gi.eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        if config_setup['add_galaxy_galaxy_lensing']:
            pk_gm_z[i, :] = pk_gm.eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        if config_setup['add_magnification']:
            pk_mi_z[i, :] = pk_mi.eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])
        if config_setup['add_galaxy_galaxy_lensing'] and config_setup['add_magnification']:
            pk_mm_z[i, :] = config_setup['pk_mm'].eval(config_setup['k'][i],1./(1+z_i), config_setup['cosmo'])

    def chunk_cl_integrals_wgp(zm_chunk_size, l_chunk_size):

        zm_chunks = [range(i, min(i + zm_chunk_size, len(config_setup['zm_centers']))) for i in range(0, len(config_setup['zm_centers']), zm_chunk_size)]
        l_chunks = [range(i, min(i + l_chunk_size, len(config_setup['l']))) for i in range(0, len(config_setup['l']), l_chunk_size)]
    
        cl_wgp = np.zeros((len(config_setup['Pi_h']), len(config_setup['zm_centers']), len(config_setup['l'])))
    
        for zm_chunk in zm_chunks:
            for l_chunk in l_chunks:
                cl_gi_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_wgp'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_gi_z[:, l_chunk])
                if config_setup['add_galaxy_galaxy_lensing']:
                    cl_gG_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_z1_lensing_z2_wgp'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_gm_z[:, l_chunk])
                if config_setup['add_magnification']:
                    cl_mi_integrand = 2*(config_setup['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['error_dist_z2_lensing_z1_wgp'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_mi_z[:, l_chunk])
                if config_setup['add_galaxy_galaxy_lensing'] and config_setup['add_magnification']:
                    cl_mG_integrand = 2*(config_setup['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config_setup['lensing_z1_lensing_z2_wgp'][:, :, zm_chunk], 1/(config_setup['cmd']**2)), pk_mm_z[:, l_chunk])

                if not config_setup['add_galaxy_galaxy_lensing'] and not config_setup['add_magnification']:
                    cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand, config_setup['cmd'], axis = 0)
                if config_setup['add_galaxy_galaxy_lensing'] and not config_setup['add_magnification']:
                    cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_gG_integrand, config_setup['cmd'], axis = 0)
                if not config_setup['add_galaxy_galaxy_lensing'] and config_setup['add_magnification']:
                    cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_mi_integrand, config_setup['cmd'], axis = 0)
                if config_setup['add_galaxy_galaxy_lensing'] and config_setup['add_magnification']:
                    cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_gG_integrand + cl_mi_integrand + cl_mG_integrand, config_setup['cmd'], axis = 0)
    
        return cl_wgp

    zm_chunk_size = 10
    l_chunk_size = 101
    cl_wgp = chunk_cl_integrals_wgp(zm_chunk_size, l_chunk_size)
    
    theta = np.einsum('i, j->ij', config_setup['rp_model'], 1/config_setup['y3fid'].comoving_distance(config_setup['zm_centers']).value)
    
    #Hankel transform
    integrand_wgp = np.einsum('ijk, ljk->iljk', config_setup['l']*scipy.special.jv(2, np.einsum('i, jk->jki', config_setup['l'], theta)), cl_wgp)/(2*np.pi)
    
    corr_function_phot_wgp = np.trapz(integrand_wgp, config_setup['l'], axis = 3)
    
    #Integration over zm
    zm_integration_phot_wgp = np.trapz(np.einsum('i,jki->jki', config_setup['kernel_wgp'], corr_function_phot_wgp), config_setup['zm_centers'], axis = 2)
    
    #Integration over pi
    wgp_phot = -np.trapz(zm_integration_phot_wgp, config_setup['Pi_h'], axis = 1)
    
    return wgp_phot.value

