import numpy as np
import pyccl as ccl

import pandas as pd
import argparse
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import c
import scipy
from scipy.interpolate import splev, splrep
import pyccl.nl_pt as pt

from compute_observables import wgg_spec_lightcone, wgp_spec_lightcone, wgg_spec_snapshot, wgp_spec_snapshot, wgg_phot_lightcone, wgp_phot_lightcone
from likelihood import read_config, run_sampler

config_setup = dict(z_min = 0., # Minimum redshift to model
                    z_max = 1.1, # Maximum redshift to model
                    z_snapshot=0, #redshift of the snapshot
                    num_k = 10001,
                    bins_z = 100, # Number of redsfhit bins
                    rp_model_min = 0.145, # Minimum transverse distance to model in Mpc
                    rp_model_max = 26.09, # Maximum transverse distance to model in Mpc
                    bins_rp_model = 18, # Number of transverse distance bins
                    log10kmin = -5, # minimum k
                    log10kmax = 2, # maximum k
                    l_min = 0, # Minimum l
                    l_max = 10001, # Maximum l
                    steps_l = 10, # Steps in l
                    H0 = 69.,
                    Om_m = 0.25, # Omega matter
                    Om_b = 0.044, # Omega baryons
                    sigma8 = 0.8,
                    n_s = 0.95,
                    #H0 = 68.1,
                    #Om_m = 0.306, # Omega matter
                    #Om_b = 0.0486, # Omega baryons
                    #sigma8 = 0.815,
                    #n_s = 0.967,
                    IA_model = 'NLA', # Model for IA
                    min_scale_cut = 2, # Minimum scale cut to apply in the correlation function in Mpc/h
                    max_scale_cut = 100, 
                    z_type = 'spec', # It can either be "phot" or "spec"
                    Pi = np.array([-233,-144,-89,-55,-34,-21,-13,-8,-5,-3,-2,-1,0,1,2,3,5,8,13,21,34,55,89,144,233])* u.Mpc/0.69, # Pi binning
                    bins_zm = 10, # Number of redshift bins for the error distribution in the phot case
                    sampler = 'nautilus', # Allows to choose between evaluate and emcee (for the moment)
                    box=False, #is it a box or a lightcone
                    n_cores = 10
                    )

def update_config(config_setup):
    # This function computes some quantities that are needed for the modelling of the observables and only need to be defined once.
    
    # I set the cosmology in astropy and CCL objects
    config_setup['y3fid'] = FlatLambdaCDM(H0=config_setup['H0'], Om0=config_setup['Om_m'], Ob0=config_setup['Om_b']) # This is defined to compute comoving distances
    config_setup['cosmo'] = ccl.Cosmology(Omega_c=config_setup['Om_m']-config_setup['Om_b'], Omega_b=config_setup['Om_b'], 
                          h=config_setup['H0']/100, sigma8 = config_setup['sigma8'], n_s=config_setup['n_s'])
    
    # I define the transverse distance
    config_setup['rp_model'] = np.logspace(np.log10(config_setup['rp_model_min']), np.log10(config_setup['rp_model_max']), config_setup['bins_rp_model'])
    config_setup['k_model'] = np.logspace(config_setup['log10kmin'], config_setup['log10kmax'], config_setup['num_k'])
    # I define the redshift, the l and the k.
    z = np.linspace(config_setup['z_min'], config_setup['z_max'], config_setup['bins_z'])
    config_setup['z_centers'] = (z[:-1]+z[1:])/2
    config_setup['cmd'] = config_setup['y3fid'].comoving_distance(config_setup['z_centers']).value
    config_setup['l'] = np.arange(config_setup['l_min'], config_setup['l_max'], config_setup['steps_l'])
    config_setup['k'] = np.array([(config_setup['l'] + 0.5) / j for j in config_setup['cmd']])
    
    def kernel_wz_spec(cat1, cat2, config_setup):

        nz_cat1, _ = np.histogram(cat1, bins = z, density = True)
        nz_cat2, _ = np.histogram(cat2, bins = z, density = True)
        diff_cmd = np.gradient(config_setup['cmd'])/np.gradient(config_setup['z_centers'])
        kernel = ((nz_cat1*nz_cat2)/((config_setup['cmd']**2)*diff_cmd)) / (np.trapz((nz_cat1*nz_cat2)/((config_setup['cmd']**2)*diff_cmd), config_setup['z_centers']))
        
        return kernel

    def kernel_wz_phot(cat1, cat2, config_setup):

        nz_cat1, _ = np.histogram(cat1, bins = zm, density = True)
        nz_cat2, _ = np.histogram(cat2, bins = zm, density = True)
        diff_cmd = np.gradient(config_setup['cmd_zm'])/np.gradient(config_setup['zm_centers'])
        kernel = ((nz_cat1*nz_cat2)/((config_setup['cmd_zm']**2)*diff_cmd)) / (np.trapz((nz_cat1*nz_cat2)/((config_setup['cmd_zm']**2)*diff_cmd), config_setup['zm_centers']))
        
        return kernel

    # I read the catalogues from positions and shapes to define the n(z) distribution in the lightcone.
    if config_setup['box'] == False:
        path_nz = '/nfs/pic.es/user/d/dnavarro/IATheory/data/nz/'
        positions_nz = pd.read_csv(path_nz + 'positions_nz.csv')
        shapes_nz = pd.read_csv(path_nz + 'shapes_nz.csv')

        if config_setup['z_type'] == 'spec':
            config_setup['kernel_wgg'] = kernel_wz_spec(positions_nz['zs'], positions_nz['zs'], config_setup)
            config_setup['kernel_wgp'] = kernel_wz_spec(positions_nz['zs'], shapes_nz['zs'], config_setup)
            
        elif config_setup['z_type'] == 'phot':
            zm = np.linspace(config_setup['z_min'], config_setup['z_max'], config_setup['bins_zm'])
            config_setup['zm_centers'] = (zm[:-1]+zm[1:])/2
            config_setup['cmd_zm'] = config_setup['y3fid'].comoving_distance(config_setup['zm_centers']).value
            config_setup['kernel_wgg'] = kernel_wz_phot(positions_nz['zb'], positions_nz['zb'], config_setup)
            config_setup['kernel_wgp'] = kernel_wz_phot(positions_nz['zb'], shapes_nz['zb'], config_setup)

    # I initialize a PyCCL object needed to compute the observables.
    config_setup['ptc_gg'] = pt.PTCalculator(with_NC=True, with_IA=False,
                          log10k_min=config_setup["log10kmin"], log10k_max=config_setup["log10kmax"], nk_per_decade=20)
    
    config_setup['ptc_gp'] = pt.PTCalculator(with_NC=True, with_IA=True,
                          log10k_min=config_setup["log10kmin"], log10k_max=config_setup["log10kmax"], nk_per_decade=20)

    if config_setup['z_type'] == 'phot':
        error_dist_measured_wgg = positions_nz.copy(deep=True) # For the moment I define the same error dist as positions
        error_dist_measured_wgp = shapes_nz.copy(deep=True) # For the moment I define the same error dist as shapes
        
        error_dist_measured_wgg['r_zb'] = config_setup['y3fid'].comoving_distance(error_dist_measured_wgg['zb']).value
        error_dist_measured_wgg['r_zs'] = config_setup['y3fid'].comoving_distance(error_dist_measured_wgg['zs']).value
        
        error_dist_measured_wgp['r_zb'] = config_setup['y3fid'].comoving_distance(error_dist_measured_wgp['zb']).value
        error_dist_measured_wgp['r_zs'] = config_setup['y3fid'].comoving_distance(error_dist_measured_wgp['zs']).value


        def compute_error_dist(cmd, zm, pi, error_dist_measured):

            z1 = zm - ((pi*config_setup['y3fid'].H(zm))/(2*c.to('km/s')))
            z2 = zm + ((pi*config_setup['y3fid'].H(zm))/(2*c.to('km/s')))
        
            z_widht = 0.05
            
            #Evaluate the error distributions
            positions_z1 = error_dist_measured[error_dist_measured.r_zb.between(config_setup['y3fid'].comoving_distance(z1-z_widht).value, config_setup['y3fid'].comoving_distance(z1+z_widht).value)]
            counts_z1, bins_z1 = np.histogram(positions_z1.r_zs, bins = 50, density = True)
            bins_z1_center = (bins_z1[:-1]+bins_z1[1:])/2
            if len(positions_z1)<40:
                counts_z1[:]=0
            spl_z1 = splrep(bins_z1_center, counts_z1)
        
            positions_z2 = error_dist_measured[error_dist_measured.r_zb.between(config_setup['y3fid'].comoving_distance(z2-z_widht).value, config_setup['y3fid'].comoving_distance(z2+z_widht).value)]
            counts_z2, bins_z2 = np.histogram(positions_z2.r_zs, bins = 50, density = True)
            bins_z2_center = (bins_z2[:-1]+bins_z2[1:])/2
            if len(positions_z2)<40:
                counts_z2[:]=0
            spl_z2 = splrep(bins_z2_center, counts_z2)
            
            lensing_kernel_z1 = np.zeros(len(cmd))
            lensing_kernel_z2 = np.zeros(len(cmd))
            cmd_to_z = splrep(cmd, config_setup['z_centers'])
            for i, cmd_i in enumerate(cmd):
                prefactor_lensing_kernel = ((3*config_setup['y3fid'].H0**2*config_setup['y3fid'].Om0)/(2*c.to('km/s')**2)).value*(cmd_i/config_setup['y3fid'].scale_factor(splev(cmd_i, cmd_to_z, ext = 1)))
                lensing_kernel_z1[i] = prefactor_lensing_kernel*np.trapz(splev(cmd, spl_z1, ext = 1)*(cmd-cmd_i)/cmd, cmd, axis = 0)
                lensing_kernel_z2[i] = prefactor_lensing_kernel*np.trapz(splev(cmd, spl_z2, ext = 1)*(cmd-cmd_i)/cmd, cmd, axis = 0)
                
            lensing_kernel_z1[lensing_kernel_z1<0] = 0.
            lensing_kernel_z2[lensing_kernel_z2<0] = 0.
        
            error_dist = (splev(cmd, spl_z1, ext = 1)*splev(cmd, spl_z2, ext = 1))
            error_dist_z1_lensing_z2 = (splev(cmd, spl_z1, ext = 1)*lensing_kernel_z2)
            error_dist_z2_lensing_z1 = (splev(cmd, spl_z2, ext = 1)*lensing_kernel_z1)
            lensing_z1_lensing_z2 = lensing_kernel_z1*lensing_kernel_z2
        
            return error_dist, error_dist_z1_lensing_z2, error_dist_z2_lensing_z1, lensing_z1_lensing_z2

        config_setup['error_dist_wgg'] = np.zeros((len(config_setup['z_centers']), len(config_setup['Pi']), len(config_setup['zm_centers'])))
        config_setup['error_dist_z1_lensing_z2_wgg'] = np.zeros_like(config_setup['error_dist_wgg'])
        config_setup['error_dist_z2_lensing_z1_wgg'] = np.zeros_like(config_setup['error_dist_wgg'])
        config_setup['lensing_z1_lensing_z2_wgg'] = np.zeros_like(config_setup['error_dist_wgg'])
        config_setup['error_dist_wgp'] = np.zeros((len(config_setup['z_centers']), len(config_setup['Pi']), len(config_setup['zm_centers'])))
        config_setup['error_dist_z1_lensing_z2_wgp'] = np.zeros_like(config_setup['error_dist_wgp'])
        config_setup['error_dist_z2_lensing_z1_wgp'] = np.zeros_like(config_setup['error_dist_wgp'])
        config_setup['lensing_z1_lensing_z2_wgp'] = np.zeros_like(config_setup['error_dist_wgp'])
        for i, Pi_i in enumerate(config_setup['Pi']):
            for j, zm_i in enumerate(config_setup['zm_centers']):
                config_setup['error_dist_wgg'][:, i, j], config_setup['error_dist_z1_lensing_z2_wgg'][:, i, j], config_setup['error_dist_z2_lensing_z1_wgg'][:, i, j], config_setup['lensing_z1_lensing_z2_wgg'][:, i, j] = compute_error_dist(config_setup['cmd'], zm_i, Pi_i, error_dist_measured_wgg)
                config_setup['error_dist_wgp'][:, i, j], config_setup['error_dist_z1_lensing_z2_wgp'][:, i, j], config_setup['error_dist_z2_lensing_z1_wgp'][:, i, j], config_setup['lensing_z1_lensing_z2_wgp'][:, i, j] = compute_error_dist(config_setup['cmd'], zm_i, Pi_i, error_dist_measured_wgp)

        # Matter
        config_setup['ptt_m'] = pt.PTMatterTracer()
        # Matter x matter
        config_setup['pk_mm'] = pt.get_pt_pk2d(config_setup['cosmo'], config_setup['ptt_m'], tracer2=config_setup['ptt_m'], ptc=config_setup['ptc_gp'])

        path_modeling_distributions = '/data/astro/scratch/dnavarro/PAUS_IA/paper/measurements/PAUS_data/modeling/'
        magnification_alpha = pd.read_parquet(path_modeling_distributions + 'magnification_alpha.pq')
        config_setup['alpha'] = magnification_alpha['bright_no_zb_cut_0_no_luminosity_cut_0_red_NUVr_BB_2_colors'].values

    return config_setup

config_setup = update_config(config_setup)

def run():

    if config_setup['IA_model'] == 'NLA':
        n_dim = 3
        aprox_bias = np.asarray([1.2, -0.4, 0.5])
    else:
        n_dim = 5
        aprox_bias = np.asarray([1.2, -0.4, 0.5, 1, 1.5])

    if config_setup['sampler'] == 'evaluate':
        if config_setup['box'] == False:
            if config_setup['z_type'] == 'spec':
                corr_model_wgg = wgg_spec_lightcone.model_wgg_spec_lightcone(aprox_bias, config_setup)
                corr_model_wgp = wgp_spec_lightcone.model_wgp_spec_lightcone(aprox_bias, config_setup)
            elif config_setup['z_type'] == 'phot':
                corr_model_wgg = wgg_phot_lightcone.model_wgg_phot_lightcone(aprox_bias, config_setup)
                corr_model_wgp = wgp_phot_lightcone.model_wgp_phot_lightcone(aprox_bias, config_setup)
            else:
                print('ERROR: z_type must be spec or phot')
        else:
            corr_model_wgg = wgg_spec_snapshot.model_wgg_spec_snapshot(aprox_bias, config_setup)
            corr_model_wgp = wgp_spec_snapshot.model_wgp_spec_snapshot(aprox_bias, config_setup)
        
        print(corr_model_wgg)
        print(corr_model_wgp)
    elif config_setup['sampler'] != 'evaluate':
        read_config.init_config(config_setup) # This function is used to pass the config information to the run_sampler.py as a global variable
        run_sampler.init_config()
        run_sampler.initialize_data()
        if config_setup['sampler'] == 'emcee':
            run_sampler.run_emcee()
        elif config_setup['sampler'] == 'nautilus':
            run_sampler.run_nautilus()
        else:
            print('Choose between emcee or nautilus')
    else:
        print('Choose between evaluate, emcee or nautilus')
    
    return None

if __name__ == '__main__':
    
    run()

