import numpy as np
import pyccl as ccl

import pandas as pd
import argparse
from astropy.cosmology import FlatLambdaCDM
import scipy
import pyccl.nl_pt as pt

from IATheory import wgg_spec_lightcone, wgp_spec_lightcone

config_setup = dict(z_min = 0., # Minimum redshift to model
                    z_max = 1.1, # Maximum redshift to model
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
                    IA_model = 'NLA', # Model for IA
                    min_scale_cut = 2, # Minimum scale cut to apply in the correlation function in Mpc/h
                    sampler = 'evaluate', # Allows to choose between evaluate and emcee (for the moment)
                    )

def update_config(config_setup):
    # This function computes some quantities that are needed for the modelling of the observables and only need to be defined once.
    
    # I set the cosmology in astropy and CCL objects
    config_setup['y3fid'] = FlatLambdaCDM(H0=config_setup['H0'], Om0=config_setup['Om_m'], Ob0=config_setup['Om_b']) # This is defined to compute comoving distances
    config_setup['cosmo'] = ccl.Cosmology(Omega_c=config_setup['Om_m']-config_setup['Om_b'], Omega_b=config_setup['Om_b'], 
                          h=config_setup['H0']/100, sigma8 = config_setup['sigma8'], n_s=config_setup['n_s'])
    
    # I define the transverse distance
    config_setup['rp_model'] = np.logspace(np.log10(config_setup['rp_model_min']), np.log10(config_setup['rp_model_max']), config_setup['bins_rp_model'])

    # I define the redshift, the l and the k.
    z = np.linspace(config_setup['z_min'], config_setup['z_max'], config_setup['bins_z'])
    config_setup['z_centers'] = (z[:-1]+z[1:])/2
    cmd = config_setup['y3fid'].comoving_distance(config_setup['z_centers']).value
    config_setup['l'] = np.arange(config_setup['l_min'], config_setup['l_max'], config_setup['steps_l'])
    config_setup['k'] = np.array([(config_setup['l'] + 0.5) / j for j in cmd])
    
    def kernel_wz(cat1, cat2, config_setup):
    
        nz_cat1, _ = np.histogram(cat1, bins = z, density = True)
        nz_cat2, _ = np.histogram(cat2, bins = z, density = True)
        diff_cmd = np.gradient(cmd)/np.gradient(config_setup['z_centers'])
    
        kernel = ((nz_cat1*nz_cat2)/((cmd**2)*diff_cmd)) / (np.trapz((nz_cat1*nz_cat2)/((cmd**2)*diff_cmd), config_setup['z_centers']))
    
        return kernel

    # I read the catalogues from positions and shapes to define the n(z) distribution in the lightcone.
    path_nz = '/nfs/pic.es/user/d/dnavarro/IATheory/data/nz/'
    positions_nz = pd.read_parquet(path_nz + 'positions_nz.pq')
    shapes_nz = pd.read_parquet(path_nz + 'shapes_nz.pq')
    
    config_setup['kernel_wgg'] = kernel_wz(positions_nz['zs'], positions_nz['zs'], config_setup)
    config_setup['kernel_wgp'] = kernel_wz(positions_nz['zs'], shapes_nz['zs'], config_setup)

    # I initialize a PyCCL object needed to compute the observables.
    config_setup['ptc_gg'] = pt.PTCalculator(with_NC=True, with_IA=False,
                          log10k_min=config_setup["log10kmin"], log10k_max=config_setup["log10kmax"], nk_per_decade=20)
    
    config_setup['ptc_gp'] = pt.PTCalculator(with_NC=True, with_IA=True,
                          log10k_min=config_setup["log10kmin"], log10k_max=config_setup["log10kmax"], nk_per_decade=20)

    return config_setup

config_setup = update_config(config_setup)

# If we want to run a likelihood, we need to read some data and then define the priors, likelihood and probability functions
if config_setup['sampler'] != 'evaluate':

    def read_data():
        # I read the data vectors and the covariance matrix.
        path_catalogues = '/nfs/pic.es/user/d/dnavarro/IATheory/data/catalogues/'
        wgg_measured = pd.read_csv(path_catalogues + 'wgg_MICE_zs.txt', sep = ' ')
        wgp_measured = pd.read_csv(path_catalogues + 'wgp_MICE_zs.txt', sep = ' ')
        cov_mat = pd.read_csv(path_catalogues + 'cov_std_MICE_zs.txt', sep = ' ', header = None)
        cov_mat.columns = np.concatenate([wgg_measured.r.values, wgg_measured.r.values])
        cov_mat.set_index(np.concatenate([wgg_measured.r.values, wgg_measured.r.values]), inplace = True)
        
        # I apply the scale cuts
        min_rp_scale = config_setup['min_scale_cut']/0.69
        wgg_measured = wgg_measured[wgg_measured.r > min_rp_scale]
        wgp_measured = wgp_measured[wgp_measured.r > min_rp_scale]
        cov_mat = cov_mat.loc[(cov_mat.columns>min_rp_scale), (cov_mat.columns>min_rp_scale)].values
        
        # I save the transverse distance and the correlation functions
        rp_data = wgg_measured.r
        corr_wgg = wgg_measured.wgg
        corr_wgp = wgp_measured.wgp
        corr = pd.concat([corr_wgg, corr_wgp])
    
        return rp_data, corr, cov_mat
    
    rp_data, corr, cov_mat = read_data()

    def log_prior(p):
        
        b_1 = p[0]
        b_2 = p[1]
        if config_setup['IA_model'] == 'NLA':
            a_1 = p[2]
            if not ((0 < b_1 < 2) & (-8 < a_1 < 8)):
                return -np.inf
        
        elif config_setup['IA_model'] == 'TATT':
            a_1 = p[2]
            a_2 = p[3]
            a_d = p[4]
            if not ((0 < b_1 < 2) & (-8 < a_1 < 8) & (-12 < a_2 < 12) & (-12 < a_d < 12)):
                return -np.inf
        
        mu = 0
        sigma = 0.5
        return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(b_2-mu)**2/sigma**2
    
    def log_likelihood(p):
        
        corr_model_wgg = wgg_spec_lightcone.model_wgg_spec_lightcone(p, config_setup)
        corr_model_wgg_interpol = np.interp(rp_data, config_setup['rp_model'], corr_model_wgg)
        
        corr_model_wgp = wgp_spec_lightcone.model_wgp_spec_lightcone(p, config_setup)
        corr_model_wgp_interpol = np.interp(rp_data, config_setup['rp_model'], corr_model_wgp)

        corr_model_interpol = np.concatenate([corr_model_wgg_interpol, corr_model_wgp_interpol])
    
        delta = (corr - corr_model_interpol)
        inv_cov = np.linalg.pinv(cov_mat)
        chisq = delta.dot(inv_cov.dot(delta.T))
        ll = -0.5 * chisq
        if np.isnan(ll):
            return -np.inf, chisq
        return ll, chisq
    
    def log_probability(p):
        
        lp = log_prior(p)
    
        if not np.isfinite(lp):
            return -np.inf, log_likelihood(p)[1]
        return lp + log_likelihood(p)[0], log_likelihood(p)[1]
    
    path_chains = '/nfs/pic.es/user/d/dnavarro/IATheory/data/chains/' # Save the chains
    filename = path_chains + "wgg_wgp_spec_{}_{}_Mpc_h.h5".format(config_setup['IA_model'], config_setup['min_scale_cut'])

def run():
    
    if config_setup['IA_model'] == 'NLA':
        n_dim = 3
        aprox_bias = np.asarray([1.2, -0.4, 0.5])
    else:
        n_dim = 5
        aprox_bias = np.asarray([1.2, -0.4, 0.5, 1, 1.5])

    if config_setup['sampler'] == 'evaluate':
        corr_model_wgg = wgg_spec_lightcone.model_wgg_spec_lightcone(aprox_bias, config_setup)
        corr_model_wgp = wgp_spec_lightcone.model_wgp_spec_lightcone(aprox_bias, config_setup)
        print(corr_model_wgg)
        print(corr_model_wgp)
    elif config_setup['sampler'] == 'emcee':
        import emcee
        from multiprocessing import Pool
        
        n_walkers = 12 #32
        n_steps = 100 #10000
        initial = aprox_bias + 0.1 * np.random.randn(n_walkers, n_dim)
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(n_walkers, n_dim)
        
        # We'll track how the average autocorrelation time estimate changes
        index = 0
        autocorr = np.empty(n_steps)
    
        # This will be useful to testing convergence
        old_tau = np.inf
        
        # I run the chains
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            log_probability,
            moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
            pool=pool,
            backend=backend
            )
            # Now we'll sample for up to n_steps
            for sample in sampler.sample(initial, iterations=n_steps, progress=True):
                # Only check convergence every 100 steps
                if sampler.iteration % 100:
                    continue
    
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
    
                # Check convergence
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
    else:
        print('Choose between evaluate or emcee')
    
    return None

if __name__ == '__main__':
    
    run()

