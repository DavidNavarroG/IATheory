import numpy as np
import pyccl.nl_pt as pt
import scipy

class model_wgg_spec_box():

    """
        Models the projected galaxy-galaxy correlation with spectroscopic redshifts in a box.
        
        Arguments:
        -----------
            config (dict): Configuration dictionary for the computation.
        Attributes:
        -----------
            xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, pk_gg):

        z_i = config['z_box']
        pk_gg_z = pk_gg(config['k_model'],1./(1+z_i), config['cosmo'])
        integrand = pk_gg_z*config['k_model']*scipy.special.j0(np.array([config['k_model']*rp_i for rp_i in config['rp_model']]))/(2*np.pi)
        self.xi = np.trapz(integrand, config['k_model'], axis = 1)

class model_wgg_spec_lightcone():

    """
    Models the projected galaxy-galaxy correlation with spectroscopic redshifts in a lightcone.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, pk_gg):

        pk_gg_z = np.zeros_like(config['k'])
        corr_function_spec = np.zeros((len(config['rp_model']), len(config['z_centers'])))
        for i, z_i in enumerate(config['z_centers']):
            pk_gg_z[i, :] = pk_gg(config['k'][i],1./(1+z_i), config['cosmo'])
            cl = pk_gg_z[i, :]/(config['y3fid'].comoving_distance(z_i).value**2)
            theta = config['rp_model'] / config['y3fid'].comoving_distance(z_i).value
            #Hankel transform
            integrand = (config['l']*scipy.special.j0(np.array([config['l'] * theta_i for theta_i in theta]))*cl)/(2*np.pi)
            corr_function_spec[:, i] = np.trapz(integrand, config['l'], axis = 1)

        #Integration over z
        self.xi = np.trapz(np.einsum('i,ji->ji', config['kernel_wgg'], corr_function_spec), config['z_centers'], axis = 1)

class model_wgg_phot_lightcone():

    """
    Models the projected galaxy-galaxy correlation with photometric redshifts in a lightcone.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, pk_gg, ptt_g):

        # I evaluate the power spectrum
        pk_gg_z = np.zeros_like(config['k'])

        if config['add_magnification']:
            # Galaxies x matter
            pk_gm = config['ptc_gg'].get_biased_pk2d(ptt_g, tracer2=config['ptt_m'])
            pk_gm_z = np.zeros_like(config['k'])
            pk_mm_z = np.zeros_like(config['k'])
            
        for i, z_i in enumerate(config['z_centers']):
            pk_gg_z[i, :] = pk_gg(config['k'][i],1./(1+z_i), config['cosmo'])
            if config['add_magnification']:
                pk_gm_z[i, :] = pk_gm(config['k'][i],1./(1+z_i), config['cosmo'])
                pk_mm_z[i, :] = config['pk_mm'](config['k'][i],1./(1+z_i), config['cosmo'])

        def chunk_cl_integrals_wgg(zm_chunk_size, l_chunk_size):

            zm_chunks = [range(i, min(i + zm_chunk_size, len(config['zm_centers']))) for i in range(0, len(config['zm_centers']), zm_chunk_size)]
            l_chunks = [range(i, min(i + l_chunk_size, len(config['l']))) for i in range(0, len(config['l']), l_chunk_size)]
        
            cl_wgg = np.zeros((len(config['Pi_h']), len(config['zm_centers']), len(config['l'])))
        
            for zm_chunk in zm_chunks:
                for l_chunk in l_chunks:
                    cl_gg_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_wgg'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_gg_z[:, l_chunk])
                    if config['add_magnification']:
                        cl_mm_integrand = 4*(config['alpha']-1)*(config['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['lensing_z1_lensing_z2_wgg'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_mm_z[:, l_chunk])
                        cl_gm_integrand = 2*(config['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_z1_lensing_z2_wgg'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_gm_z[:, l_chunk])
                        cl_mg_integrand = 2*(config['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_z2_lensing_z1_wgg'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_gm_z[:, l_chunk])
                        cl_wgg[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gg_integrand+cl_mm_integrand+cl_gm_integrand+cl_mg_integrand, config['cmd'], axis = 0)
                    else:
                        cl_wgg[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gg_integrand, config['cmd'], axis = 0)
        
            return cl_wgg

        zm_chunk_size = 10
        l_chunk_size = 101
        cl_wgg = chunk_cl_integrals_wgg(zm_chunk_size, l_chunk_size)
        theta = np.einsum('i, j->ij', config['rp_model'], 1/config['y3fid'].comoving_distance(config['zm_centers']).value)
        
        #Hankel transform
        integrand_wgg = np.einsum('ijk, ljk->iljk', config['l']*scipy.special.j0(np.einsum('i, jk->jki', config['l'], theta)), cl_wgg)/(2*np.pi)
        
        corr_function_phot_wgg = np.trapz(integrand_wgg, config['l'], axis = 3)
        
        #Integration over zm
        zm_integration_phot_wgg = np.trapz(np.einsum('i,jki->jki', config['kernel_wgg'], corr_function_phot_wgg), config['zm_centers'], axis = 2)
        
        #Integration over pi
        self.xi = np.trapz(zm_integration_phot_wgg, config['Pi_h'], axis = 1)

class model_wgp_spec_box():

    """
        Models the projected galaxy-intrinsic correlation with spectroscopic redshifts in a box.
        
        Arguments:
        -----------
            config (dict): Configuration dictionary for the computation.
        Attributes:
        -----------
            xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, ptt_g, a_1, a_2, a_d):

        z_i = config['z_box']
        # Let's convert the a_IA values into the correctly normalized c_IA values
        c_1,c_d,c_2 = pt.translate_IA_norm(config['cosmo'], z=z_i, a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)
        
        # Intrinsic alignments
        if config['IA_model'] == 'NLA':
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=c_1)
        else:
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=c_1, c2=c_2, cdelta=c_d)
        
        pk_gi = config['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=ptt_i)
        
        pk_gi_z = pk_gi(config['k_model'],1./(1+z_i), config['cosmo'])
        integrand = pk_gi_z*config['k_model']*scipy.special.jv(2, np.array([config['k_model']*rp_i for rp_i in config['rp_model']]))/(2*np.pi)
        self.xi = -np.trapz(integrand, config['k_model'], axis = 1)

class model_wgp_spec_lightcone():

    """
    Models the projected galaxy-intrinsic correlation with spectroscopic redshifts in a lightcone.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, ptt_g, a_1, a_2, a_d):
        
        # Let's convert the a_IA values into the correctly normalized c_IA values:
        c_1,c_d,c_2 = pt.translate_IA_norm(config['cosmo'], z=config['z_centers'], a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)
        
        # Intrinsic alignments
        if config['IA_model'] == 'NLA':
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config['z_centers'],c_1)) # NLA model
        else:
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config['z_centers'],c_1), c2=(config['z_centers'],c_2), cdelta=(config['z_centers'],c_d)) # TATT model

        # Calculate some power spectra with FAST-PT
        # Galaxies x galaxies.
        pk_gi = config['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=ptt_i)

        pk_gi_z = np.zeros_like(config['k'])
        corr_function_spec = np.zeros((len(config['rp_model']), len(config['z_centers'])))
        for i, z_i in enumerate(config['z_centers']):
            pk_gi_z[i, :] = pk_gi(config['k'][i],1./(1+z_i), config['cosmo'])
            cl = pk_gi_z[i, :]/(config['y3fid'].comoving_distance(z_i).value**2)
            theta = config['rp_model'] / config['y3fid'].comoving_distance(z_i).value
            #Hankel transform
            integrand = (config['l']*scipy.special.jv(2, np.array([config['l'] * theta_i for theta_i in theta]))*cl)/(2*np.pi)
            corr_function_spec[:, i] = np.trapz(integrand, config['l'], axis = 1)

        #Integration over z
        self.xi = -np.trapz(np.einsum('i,ji->ji', config['kernel_wgp'], corr_function_spec), config['z_centers'], axis = 1)

class model_wgp_phot_lightcone():

    """
    Models the projected galaxy-intrinsic correlation with photometric redshifts in a lightcone.
    
    Arguments:
    -----------
        config (dict): Configuration dictionary for the computation.
    Attributes:
    -----------
        xi (ndarray): 2D Correlation function in bins of projected and l.o.s distance.
    """
    
    def __init__(self, config, ptt_g, a_1, a_2, a_d):
        
        # Let's convert the a_IA values into the correctly normalized c_IA values:
        c_1,c_d,c_2 = pt.translate_IA_norm(config['cosmo'], z=config['z_centers'], a1=a_1, a1delta=a_d, a2=a_2, Om_m2_for_c2 = False)
        
        # Intrinsic alignments
        if config['IA_model'] == 'NLA':
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config['z_centers'],c_1)) # NLA model
        else:
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(config['z_centers'],c_1), c2=(config['z_centers'],c_2), cdelta=(config['z_centers'],c_d)) # TATT model
        
        # Calculate some power spectra with FAST-PT
        # Galaxies x intrinsic.
        pk_gi = config['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=ptt_i)
        pk_gi_z = np.zeros_like(config['k'])

        if config['add_galaxy_galaxy_lensing']:
            # Galaxies x matter
            pk_gm = config['ptc_gp'].get_biased_pk2d(ptt_g, tracer2=config['ptt_m'])
            pk_gm_z = np.zeros_like(config['k'])

        if config['add_magnification']:
            # Matter x intrinsic
            pk_mi = config['ptc_gp'].get_biased_pk2d(config['ptt_m'], tracer2=ptt_i)
            pk_mi_z = np.zeros_like(config['k'])

        if config['add_galaxy_galaxy_lensing'] and config['add_magnification']:
            pk_mm_z = np.zeros_like(config['k'])
            
        for i, z_i in enumerate(config['z_centers']):
            pk_gi_z[i, :] = pk_gi(config['k'][i],1./(1+z_i), config['cosmo'])
            if config['add_galaxy_galaxy_lensing']:
                pk_gm_z[i, :] = pk_gm(config['k'][i],1./(1+z_i), config['cosmo'])
            if config['add_magnification']:
                pk_mi_z[i, :] = pk_mi(config['k'][i],1./(1+z_i), config['cosmo'])
            if config['add_galaxy_galaxy_lensing'] and config['add_magnification']:
                pk_mm_z[i, :] = config['pk_mm'](config['k'][i],1./(1+z_i), config['cosmo'])

        def chunk_cl_integrals_wgp(zm_chunk_size, l_chunk_size):

            zm_chunks = [range(i, min(i + zm_chunk_size, len(config['zm_centers']))) for i in range(0, len(config['zm_centers']), zm_chunk_size)]
            l_chunks = [range(i, min(i + l_chunk_size, len(config['l']))) for i in range(0, len(config['l']), l_chunk_size)]
        
            cl_wgp = np.zeros((len(config['Pi_h']), len(config['zm_centers']), len(config['l'])))
        
            for zm_chunk in zm_chunks:
                for l_chunk in l_chunks:
                    cl_gi_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_wgp'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_gi_z[:, l_chunk])
                    if config['add_galaxy_galaxy_lensing']:
                        cl_gG_integrand = np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_z1_lensing_z2_wgp'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_gm_z[:, l_chunk])
                    if config['add_magnification']:
                        cl_mi_integrand = 2*(config['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['error_dist_z2_lensing_z1_wgp'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_mi_z[:, l_chunk])
                    if config['add_galaxy_galaxy_lensing'] and config['add_magnification']:
                        cl_mG_integrand = 2*(config['alpha']-1)*np.einsum('ijk,il->ijkl', np.einsum('ijk,i->ijk', config['lensing_z1_lensing_z2_wgp'][:, :, zm_chunk], 1/(config['cmd']**2)), pk_mm_z[:, l_chunk])

                    if not config['add_galaxy_galaxy_lensing'] and not config['add_magnification']:
                        cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand, config['cmd'], axis = 0)
                    if config['add_galaxy_galaxy_lensing'] and not config['add_magnification']:
                        cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_gG_integrand, config['cmd'], axis = 0)
                    if not config['add_galaxy_galaxy_lensing'] and config['add_magnification']:
                        cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_mi_integrand, config['cmd'], axis = 0)
                    if config['add_galaxy_galaxy_lensing'] and config['add_magnification']:
                        cl_wgp[:, zm_chunk.start:zm_chunk.stop, l_chunk.start:l_chunk.stop] = np.trapz(cl_gi_integrand + cl_gG_integrand + cl_mi_integrand + cl_mG_integrand, config['cmd'], axis = 0)
        
            return cl_wgp

        zm_chunk_size = 10
        l_chunk_size = 101
        cl_wgp = chunk_cl_integrals_wgp(zm_chunk_size, l_chunk_size)
        
        theta = np.einsum('i, j->ij', config['rp_model'], 1/config['y3fid'].comoving_distance(config['zm_centers']).value)
        
        #Hankel transform
        integrand_wgp = np.einsum('ijk, ljk->iljk', config['l']*scipy.special.jv(2, np.einsum('i, jk->jki', config['l'], theta)), cl_wgp)/(2*np.pi)
        
        corr_function_phot_wgp = np.trapz(integrand_wgp, config['l'], axis = 3)
        
        #Integration over zm
        zm_integration_phot_wgp = np.trapz(np.einsum('i,jki->jki', config['kernel_wgp'], corr_function_phot_wgp), config['zm_centers'], axis = 2)
        
        #Integration over pi
        self.xi = -np.trapz(zm_integration_phot_wgp, config['Pi_h'], axis = 1)

class model_2p_corr():
    """
    This class computes projected galaxy-galaxy (wgg) and galaxy-shear (wg+) correlation functions.
    It also allows to compute these observables in a lightcone and in a box.
    In the case of a lightcone, it can model spectroscopic and photometric redshifts cases.

    Arguments:
    -----------
        galaxy_bias (nunmpy array): Galaxy bias values
        ia_params (nunmpy array): Intrinsic alignment values

    Attributes:
    -----------
        config (dict): Configuration dictionary for the computation.
        wgg_spec_box (object, optional): Projected galaxy-galaxy (position-position) correlation function with spectroscopic redshifts in a box.
        wgg_spec_lightcone (object, optional): Projected galaxy-galaxy (position-position) correlation function with spectroscopic redshifts in a lightcone.
        wgg_phot_lightcone (object, optional): Projected galaxy-galaxy (position-position) correlation function with photometric redshifts in a lightcone.
        wgp_spec_box (object, optional): Projected galaxy-intrinsic (position-shape) correlation function with spectroscopic redshifts in a box.


    Methods:
    -----------
        model_wgg_spec_box():
            Models the projected galaxy-galaxy correlation with spectroscopic redshifts in a box.
        model_wgg_spec_lightcone():
            Models the projected galaxy-galaxy correlation with spectroscopic redshifts in a lightcone.
        model_wgg_phot_lightcone():
            Models the projected galaxy-galaxy correlation with photometric redshifts in a lightcone.
        model_wgp_spec_box():
            Models the projected galaxy-intrinsic correlation with spectroscopic redshifts in a box.
        model_wgp_spec_lightcone():
            Models the projected galaxy-intrinsic correlation with spectroscopic redshifts in a lightcone.
        model_wgp_phot_lightcone():
            Models the projected galaxy-intrinsic correlation with photometric redshifts in a lightcone.
                    
    """
    def __init__(self,config, galaxy_bias, ia_params):
        
        self.config = config
        # Galaxy biases   
        self.b_1 = galaxy_bias[0]
        self.b_2 = galaxy_bias[1]
        self.b_s = (-4/7)*(self.b_1 - 1)
        self.b_3nl = self.b_1 - 1

        # Biases for IAs.
        if config['IA_model'] == 'NLA':
            self.a_1 = ia_param[0]
            self.a_2 = 0.0
            self.a_d = 0.0
        else:   
            self.a_1 = ia_params[0]
            self.a_2 = ia_params[1]
            self.a_d = ia_params[2]

        # Number counts (galaxy clustering)
        self.ptt_g = pt.PTNumberCountsTracer(b1=self.b_1, b2=self.b_2, bs=self.b_s, b3nl = self.b_3nl)

        # Calculate some power spectra with FAST-PT
        # Galaxies x galaxies.
        self.pk_gg = config['ptc_gg'].get_biased_pk2d(self.ptt_g)
    
    def model_wgg_spec_box(self, config_specific):
        self.wgg_spec_box = model_wgg_spec_box(config_specific, self.pk_gg)
    
    def model_wgg_spec_lightcone(self, config_specific):
        self.wgg_spec_lightcone = model_wgg_spec_lightcone(config_specific, self.pk_gg)
    
    def model_wgg_phot_lightcone(self, config_specific):
        self.wgg_phot_lightcone = model_wgg_phot_lightcone(config_specific, self.pk_gg, self.ptt_g)
    
    def model_wgp_spec_box(self, config_specific):
        self.wgp_spec_box = model_wgp_spec_box(config_specific, self.ptt_g, self.a_1, self.a_2, self.a_d)
    
    def model_wgp_spec_lightcone(self, config_specific):
        self.wgp_spec_lightcone = model_wgp_spec_lightcone(config_specific, self.ptt_g, self.a_1, self.a_2, self.a_d)
    
    def model_wgp_phot_lightcone(self, config_specific):
        self.wgp_phot_lightcone = model_wgp_phot_lightcone(config_specific, self.ptt_g, self.a_1, self.a_2, self.a_d)
