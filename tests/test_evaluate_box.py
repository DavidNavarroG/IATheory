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
                        72.36891191,  87.52311773, 105.85064686, 128.016 ])
WGG_GOLDEN = np.array([35.66721734, 31.08276448, 26.68610772, 22.51591288, 18.6225348 ,
                       15.06532815, 11.89307982,  9.13356375,  6.79061404,  4.81864172,
                        3.3050134 ,  2.14445542,  1.3085149 ,  0.71223553,  0.3713007 ,
                        0.25908425])
WGP_GOLDEN = np.array([0.30783545, 0.25137683, 0.20218906, 0.16004246, 0.1246772 ,
                       0.09560415, 0.07220967, 0.05384103, 0.0397543 , 0.02957744,
                       0.02148136, 0.01550635, 0.01097536, 0.00789426, 0.00526109,
                       0.00299996])

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

    case = 'box'
    config_setup_box = dict(z_box=0)
    config_box = build_specific_config(cfg, config_setup_box, case)
    model.model_wgg_spec_box(config_box)
    model.model_wgp_spec_box(config_box)

    assert model.wgg_spec_box.xi.shape == model.wgp_spec_box.xi.shape, "wgg and wgp shapes do not match"

    # Compare arrays using allclose
    assert np.allclose(cfg['rp_model'], RP_GOLDEN, rtol=1e-3, atol=1e-8), "rp does not match golden reference"
    assert np.allclose(model.wgg_spec_box.xi, WGG_GOLDEN, rtol=1e-3, atol=1e-8), "wgg does not match golden reference"
    assert np.allclose(model.wgp_spec_box.xi, WGP_GOLDEN, rtol=1e-3, atol=1e-8), "wgp does not match golden reference"
