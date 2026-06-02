import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
import pyccl as ccl
import pyccl.nl_pt as pt
from scipy.interpolate import splev, splrep
from astropy.constants import c

def init_cosmology(cfg):
    # I set the cosmology in astropy and CCL objects
    cfg['y3fid'] = FlatLambdaCDM(H0=cfg['H0'], Om0=cfg['Om_m'], Ob0=cfg['Om_b']) # This is defined to compute comoving distances
    cfg['cosmo'] = ccl.Cosmology(
        Omega_c=cfg['Om_m'] - cfg['Om_b'],
        Omega_b=cfg['Om_b'],
        h=cfg['H0'] / 100,
        sigma8=cfg['sigma8'],
        n_s=cfg['n_s'],
        T_CMB=2.725,
        Neff=3.046,
    )

def init_grids(cfg):
    cfg['rp_model'] = np.logspace(
        np.log10(cfg['rp_model_min']),
        np.log10(cfg['rp_model_max']),
        cfg['bins_rp_model'],
    )
    cfg['k_model'] = np.logspace(cfg['log10kmin'], cfg['log10kmax'], cfg['num_k'])
    cfg['l'] = np.arange(cfg['l_min'], cfg['l_max'], cfg['steps_l'])

def init_pt_calculators(cfg):
    common_kw = dict(
        log10k_min=cfg['log10kmin'],
        log10k_max=cfg['log10kmax'],
        nk_per_decade=20,
    )
    cfg['ptc_gg'] = pt.ept.EulerianPTCalculator(with_NC=True, with_IA=False, **common_kw)
    cfg['ptc_gp'] = pt.ept.EulerianPTCalculator(with_NC=True, with_IA=True, **common_kw)
    cfg['ptc_gg'].update_ingredients(cfg['cosmo'])
    cfg['ptc_gp'].update_ingredients(cfg['cosmo'])

def compute_kernels_spec(cfg, z):
    def kernel(cat1, cat2):
        nz1, _ = np.histogram(cat1, bins=z, density=True)
        nz2, _ = np.histogram(cat2, bins=z, density=True)
        diff_cmd = np.gradient(cfg['cmd']) / np.gradient(cfg['z_centers'])
        raw = (nz1 * nz2) / (cfg['cmd'] ** 2 * diff_cmd)
        return raw / np.trapz(raw, cfg['z_centers'])

    cfg['kernel_wgg'] = kernel(cfg['_positions_nz']['zs'], cfg['_positions_nz']['zs'])
    cfg['kernel_wgp'] = kernel(cfg['_positions_nz']['zs'], cfg['_shapes_nz']['zs'])

def compute_kernels_phot(cfg, zm):
    def kernel_wz_phot(cat1, cat2, cfg):
        nz1, _ = np.histogram(cat1, bins = zm, density = True)
        nz2, _ = np.histogram(cat2, bins = zm, density = True)
        diff_cmd = np.gradient(cfg['cmd_zm'])/np.gradient(cfg['zm_centers'])
        raw = ((nz1*nz2)/((cfg['cmd_zm']**2)*diff_cmd)) 
        return raw / (np.trapz(raw, cfg['zm_centers']))

    cfg['kernel_wgg'] = kernel_wz_phot(cfg['_positions_nz']['zb'], cfg['_positions_nz']['zb'], cfg)
    cfg['kernel_wgp'] = kernel_wz_phot(cfg['_positions_nz']['zb'], cfg['_shapes_nz']['zb'], cfg)

def init_lightcone(cfg):
    """Quantities shared by both spec and phot lightcones."""
    z = np.linspace(cfg['z_min'], cfg['z_max'], cfg.get('bins_z', 100))
    cfg['z_centers'] = (z[:-1] + z[1:]) / 2
    cfg['cmd'] = cfg['y3fid'].comoving_distance(cfg['z_centers']).value
    cfg['k'] = np.array([(cfg['l'] + 0.5) / d for d in cfg['cmd']])

    # Load n(z) catalogues once
    cfg['_positions_nz'] = pd.read_csv(cfg['path_nz_positions'])
    cfg['_shapes_nz'] = pd.read_csv(cfg['path_nz_shapes'])

    if cfg['z_type'] == 'spec':
        compute_kernels_spec(cfg, z)

def compute_error_distributions(cfg):

    error_dist_measured_wgg = cfg['_positions_nz'].copy(deep=True) # For the moment I define the same error dist as positions
    error_dist_measured_wgp = cfg['_shapes_nz'].copy(deep=True) # For the moment I define the same error dist as shapes
    
    error_dist_measured_wgg['r_zb'] = cfg['y3fid'].comoving_distance(error_dist_measured_wgg['zb']).value
    error_dist_measured_wgg['r_zs'] = cfg['y3fid'].comoving_distance(error_dist_measured_wgg['zs']).value
    
    error_dist_measured_wgp['r_zb'] = cfg['y3fid'].comoving_distance(error_dist_measured_wgp['zb']).value
    error_dist_measured_wgp['r_zs'] = cfg['y3fid'].comoving_distance(error_dist_measured_wgp['zs']).value


    def compute_error_dist(cmd, zm, pi, error_dist_measured):

        z1 = zm - ((pi*cfg['y3fid'].H(zm).value)/(2*c.to('km/s').value))
        z2 = zm + ((pi*cfg['y3fid'].H(zm).value)/(2*c.to('km/s').value))
    
        z_widht = 0.05
        
        #Evaluate the error distributions
        positions_z1 = error_dist_measured[error_dist_measured.r_zb.between(cfg['y3fid'].comoving_distance(z1-z_widht).value, cfg['y3fid'].comoving_distance(z1+z_widht).value)]
        counts_z1, bins_z1 = np.histogram(positions_z1.r_zs, bins = 50, density = True)
        bins_z1_center = (bins_z1[:-1]+bins_z1[1:])/2
        if len(positions_z1)<40:
            counts_z1[:]=0
        spl_z1 = splrep(bins_z1_center, counts_z1)
    
        positions_z2 = error_dist_measured[error_dist_measured.r_zb.between(cfg['y3fid'].comoving_distance(z2-z_widht).value, cfg['y3fid'].comoving_distance(z2+z_widht).value)]
        counts_z2, bins_z2 = np.histogram(positions_z2.r_zs, bins = 50, density = True)
        bins_z2_center = (bins_z2[:-1]+bins_z2[1:])/2
        if len(positions_z2)<40:
            counts_z2[:]=0
        spl_z2 = splrep(bins_z2_center, counts_z2)
        
        lensing_kernel_z1 = np.zeros(len(cmd))
        lensing_kernel_z2 = np.zeros(len(cmd))
        cmd_to_z = splrep(cmd, cfg['z_centers'])
        for i, cmd_i in enumerate(cmd):
            prefactor_lensing_kernel = ((3*cfg['y3fid'].H0**2*cfg['y3fid'].Om0)/(2*c.to('km/s')**2)).value*(cmd_i/cfg['y3fid'].scale_factor(splev(cmd_i, cmd_to_z, ext = 1)))
            lensing_kernel_z1[i] = prefactor_lensing_kernel*np.trapz(splev(cmd, spl_z1, ext = 1)*(cmd-cmd_i)/cmd, cmd, axis = 0)
            lensing_kernel_z2[i] = prefactor_lensing_kernel*np.trapz(splev(cmd, spl_z2, ext = 1)*(cmd-cmd_i)/cmd, cmd, axis = 0)
            
        lensing_kernel_z1[lensing_kernel_z1<0] = 0.
        lensing_kernel_z2[lensing_kernel_z2<0] = 0.
    
        error_dist = (splev(cmd, spl_z1, ext = 1)*splev(cmd, spl_z2, ext = 1))
        error_dist_z1_lensing_z2 = (splev(cmd, spl_z1, ext = 1)*lensing_kernel_z2)
        error_dist_z2_lensing_z1 = (splev(cmd, spl_z2, ext = 1)*lensing_kernel_z1)
        lensing_z1_lensing_z2 = lensing_kernel_z1*lensing_kernel_z2
    
        return error_dist, error_dist_z1_lensing_z2, error_dist_z2_lensing_z1, lensing_z1_lensing_z2

    cfg['error_dist_wgg'] = np.zeros((len(cfg['z_centers']), len(cfg['Pi']), len(cfg['zm_centers'])))
    cfg['error_dist_z1_lensing_z2_wgg'] = np.zeros_like(cfg['error_dist_wgg'])
    cfg['error_dist_z2_lensing_z1_wgg'] = np.zeros_like(cfg['error_dist_wgg'])
    cfg['lensing_z1_lensing_z2_wgg'] = np.zeros_like(cfg['error_dist_wgg'])
    cfg['error_dist_wgp'] = np.zeros((len(cfg['z_centers']), len(cfg['Pi']), len(cfg['zm_centers'])))
    cfg['error_dist_z1_lensing_z2_wgp'] = np.zeros_like(cfg['error_dist_wgp'])
    cfg['error_dist_z2_lensing_z1_wgp'] = np.zeros_like(cfg['error_dist_wgp'])
    cfg['lensing_z1_lensing_z2_wgp'] = np.zeros_like(cfg['error_dist_wgp'])
    for i, Pi_i in enumerate(cfg['Pi']):
        for j, zm_i in enumerate(cfg['zm_centers']):
            cfg['error_dist_wgg'][:, i, j], cfg['error_dist_z1_lensing_z2_wgg'][:, i, j], cfg['error_dist_z2_lensing_z1_wgg'][:, i, j], cfg['lensing_z1_lensing_z2_wgg'][:, i, j] = compute_error_dist(cfg['cmd'], zm_i, Pi_i, error_dist_measured_wgg)
            cfg['error_dist_wgp'][:, i, j], cfg['error_dist_z1_lensing_z2_wgp'][:, i, j], cfg['error_dist_z2_lensing_z1_wgp'][:, i, j], cfg['lensing_z1_lensing_z2_wgp'][:, i, j] = compute_error_dist(cfg['cmd'], zm_i, Pi_i, error_dist_measured_wgp)

def init_photometric(cfg):
    """Everything specific to photometric redshifts."""
    zm = np.linspace(cfg['z_min'], cfg['z_max'], cfg['bins_zm'])
    cfg['zm_centers'] = (zm[:-1] + zm[1:]) / 2
    cfg['cmd_zm'] = cfg['y3fid'].comoving_distance(cfg['zm_centers']).value

    # Kernels
    compute_kernels_phot(cfg, zm)

    # Error distributions (your existing loop logic)
    compute_error_distributions(cfg)

    # Matter tracers for lensing
    cfg['ptt_m'] = pt.PTMatterTracer()
    cfg['pk_mm'] = cfg['ptc_gp'].get_biased_pk2d(cfg['ptt_m'], tracer2=cfg['ptt_m'])

def build_specific_config(config: dict, config_specific: dict, case: str) -> dict:
    """
    case: 'box' | 'lightcone_spec' | 'lightcone_phot'
    """
    config = config.copy()
    config.update(config_specific)

    # Case-specific setup
    if not case =='box':
        init_lightcone(config)
        if config['z_type'] == 'phot':
            init_photometric(config)

    return config

def update_global_config(config_global):

    config = {**config_global}

    # Common derived quantities
    init_cosmology(config)
    init_grids(config)
    init_pt_calculators(config)

    return config