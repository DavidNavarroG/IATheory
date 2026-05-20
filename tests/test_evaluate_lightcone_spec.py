"""
Tests for the evaluation mode of IATheory.

Run by using pytest -v -W ignore::DeprecationWarning in the root directory


This module validates that `run_module.run()` produces:
- Matched shapes between wgg and wgp
- Numerical outputs identical to a known golden reference for rp, wgg, wgp

The golden arrays correspond to a fixed cosmology and IA configuration
used for regression testing. Any change to the physics, numerical integration,
or model logic that affects these outputs will cause this test to fail.
"""

import pytest
import numpy as np
from IATheory.process_config import init_cosmology, init_grids, init_pt_calculators, compute_kernels_spec, init_lightcone, init_photometric, update_global_config, build_specific_config
from IATheory.compute_observables import model_2p_corr

# Golden reference arrays
RP_GOLDEN = np.array([  7.391     ,   8.9386913 ,  10.81047248,  13.07420868,
                        15.8119761 ,  19.1230379 ,  23.12744318,  27.97037953,
                        33.82743717,  40.91097528,  49.4778215 ,  59.83858375,
                        72.36891191,  87.52311773, 105.85064686, 128.016])
WGG_GOLDEN = np.array([2.41913183e+01, 2.10380088e+01, 1.80229406e+01, 1.51755390e+01,
                       1.25338034e+01, 1.01164023e+01, 7.95693491e+00, 6.08359944e+00,
                       4.49442494e+00, 3.19209188e+00, 2.15407281e+00, 1.35616457e+00,
                       7.76346240e-01, 3.73965850e-01, 1.09482756e-01, 5.94989756e-03])
WGP_GOLDEN = np.array([0.17623536, 0.14541509, 0.11841426, 0.09505266, 0.0751753 ,
                       0.05870057, 0.04530239, 0.03455576, 0.02614209, 0.01958033,
                       0.01462287, 0.01088884, 0.00802522, 0.00587818, 0.00433258,
                       0.00289635])

def test_golden_values(reset_config):
    """

    Verify that IATheory's evaluation mode matches the golden reference output.

    This test:
    - Sets up a known cosmology, IA model, and modeling scale configuration
    - Runs `run_module.run()` in `evaluate` mode with `box=True`
    - Ensures wgg and wgp have identical shapes
    - Compares rp, wgg, and wgp against pre-computed golden values using `allclose`

    The golden arrays encode the expected results for this fixed configuration.
    Any deviation indicates a regression in physics modeling, numerical accuracy,
    or configuration handling.

    """
    cfg = reset_config
    cfg.update({
        'H0': 68.1,
        'Om_m': 0.306,
        'Om_b': 0.0486,
        'sigma8': 0.807,
        'n_s': 0.967,
        'rp_model_min': 7.391,
        'rp_model_max': 128.016,
        'bins_rp_model': 16,
        'log10kmin': -5,
        'log10kmax': 2,
        'l_min': 0,
        'l_max': 10001,
        'steps_l': 10,
        'IA_model': 'TATT',
        'min_scale_cut': 5,
        'max_scale_cut': 100
    })

    cfg = update_global_config(cfg)

    galaxy_bias = [1.2, -0.4]
    ia_params = [0.5, 1, 1.5]
    model = model_2p_corr(cfg, galaxy_bias, ia_params)

    case = 'lightcone_spec'
    config_setup_lightcone_spec = dict(z_min = 0., z_max = 1.1, z_type = 'spec')
    config_lightcone_spec = build_specific_config(cfg, config_setup_lightcone_spec, case)

    model.model_wgg_spec_lightcone(config_lightcone_spec)
    model.model_wgp_spec_lightcone(config_lightcone_spec)
    
    assert model.wgg_spec_lightcone.xi.shape == model.wgp_spec_lightcone.xi.shape, "wgg and wgp shapes do not match"

    # Compare arrays using allclose
    assert np.allclose(cfg['rp_model'], RP_GOLDEN, rtol=1e-3, atol=1e-8), "rp does not match golden reference"
    assert np.allclose(model.wgg_spec_lightcone.xi, WGG_GOLDEN, rtol=1e-3, atol=1e-8), "wgg does not match golden reference"
    assert np.allclose(model.wgp_spec_lightcone.xi, WGP_GOLDEN, rtol=1e-3, atol=1e-8), "wgp does not match golden reference"
