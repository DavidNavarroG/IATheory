import numpy as np
import pandas as pd

from IATheory.compute_observables import wgg_spec_lightcone, wgp_spec_lightcone, wgg_spec_snapshot, wgp_spec_snapshot, wgg_phot_lightcone, wgp_phot_lightcone
from IATheory.read_data import read_data_mice, read_data_flamingo

from . import read_config

def init_config():
    """Initialize the module-global config variable."""
    global config_setup
    config_setup = read_config.get_config()

def initialize_data():
    """Initialize rp_data, corr, cov_mat after config is ready."""
    global rp_data, corr, cov_mat
    if config_setup['box'] == False:
        rp_data, corr, cov_mat = read_data_mice.read_data_mice(config_setup)
    else:
        rp_data, corr, cov_mat = read_data_flamingo.read_data_flamingo(config_setup)

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

    if config_setup['box'] == False:

        if config_setup['z_type'] == 'spec':
            corr_model_wgg = wgg_spec_lightcone.model_wgg_spec_lightcone(p, config_setup)
            corr_model_wgp = wgp_spec_lightcone.model_wgp_spec_lightcone(p, config_setup)
        elif config_setup['z_type'] == 'phot':
            corr_model_wgg = wgg_phot_lightcone.model_wgg_phot_lightcone(p, config_setup)
            corr_model_wgp = wgp_phot_lightcone.model_wgp_phot_lightcone(p, config_setup)
        else:
            print('ERROR: z_type must be spec or phot')
    else:
        corr_model_wgg = wgg_spec_snapshot.model_wgg_spec_snapshot(p, config_setup)
        corr_model_wgp = wgp_spec_snapshot.model_wgp_spec_snapshot(p, config_setup)

    
    corr_model_wgg_interpol = np.interp(rp_data, config_setup['rp_model'], corr_model_wgg)
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

def loglikelihood_fn(p_dict):
    # p_dict keys follow prior_obj keys order
    if config_setup['IA_model'] == 'NLA':
        param_names = ['b1', 'b2', 'a1']
    else:
        param_names = ['b1', 'b2', 'a1', 'a2', 'ad']

    p = np.array([p_dict[key] for key in param_names])
    logl, _ = log_likelihood(p)
    return logl

def run_emcee():
    import emcee
    from multiprocessing import Pool
    
    if config_setup['IA_model'] == 'NLA':
        n_dim = 3
        aprox_bias = np.asarray([1.2, -0.4, 0.5])
    else:
        n_dim = 5
        aprox_bias = np.asarray([1.2, -0.4, 0.5, 1, 1.5])

    path_chains = '/nfs/pic.es/user/d/dnavarro/IATheory/data/chains/' # Save the chains
    filename = path_chains + "wgg_wgp_{}_{}_{}_Mpc_h_emcee.h5".format(config_setup['z_type'] ,config_setup['IA_model'], config_setup['min_scale_cut'])
    
    n_walkers = 32
    n_steps = 10000

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
    
    return None

def run_nautilus():
    from nautilus import Sampler, Prior

    prior_obj = Prior()  

    if config_setup['IA_model'] == 'NLA':
        
        prior_obj.add_parameter('b1', dist=(0, 3.0))
        prior_obj.add_parameter('b2', dist=(-5.0, 5.0))
        prior_obj.add_parameter('a1', dist=(0.0, 10.0))
    else:
        
        prior_obj.add_parameter('b1', dist=(0, 5.0))
        prior_obj.add_parameter('b2', dist=(-5.0, 5.0))
        prior_obj.add_parameter('a1', dist=(-50.0, 50.0))
        prior_obj.add_parameter('a2', dist=(-50.0, 50.0))
        prior_obj.add_parameter('ad', dist=(-50.0, 50.0))

    n_live = 5000

    output_path = '/disks/shear16/herle/models/IATheory/'

    filename = output_path + f"nautilus_chain_2.h5"
    
    sampler = Sampler(
        prior_obj,
        loglikelihood_fn,
        n_live=n_live,
        pool= config_setup['n_cores'] if config_setup['n_cores'] > 1 else None,
        filepath=filename,
        resume=False,
        n_networks=16,
    )
    sampler.run(verbose=True)
    points, log_w, log_l = sampler.posterior()
    weights = np.exp(log_w)
    logz = sampler.log_z
    output_file = (output_path + f"nautilus_chain_2.npz")

    np.savez(output_file,
            samples=points,
            logl=log_l,
            weights=weights,
            logz=logz)
    
    print(f"Saved results to {output_file}")

    return None
