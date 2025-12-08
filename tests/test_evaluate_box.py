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
from IATheory import run as run_module
from IATheory.run import update_config

# Golden reference arrays
RP_GOLDEN = np.array([  7.391,        8.9386913,   10.81047248,  13.07420868,  15.8119761,
                        19.1230379,   23.12744318,  27.97037953,  33.82743717,  40.91097528,
                        49.4778215,   59.83858375,  72.36891191,  87.52311773, 105.85064686,
                        128.016])
WGG_GOLDEN = np.array([35.66722259, 31.08275703, 26.68610924, 22.51591381, 18.62253497, 15.06532796,
                       11.89307975,  9.13356402,  6.79061793,  4.8187007,   3.30488599,  2.14440029,
                        1.30851238,  0.71221128,  0.37128253,  0.25912664])
WGP_GOLDEN = np.array([0.30783555, 0.25137702, 0.20218899, 0.16004244, 0.1246772,  0.09560415,
                       0.07220968, 0.05384102, 0.03975412, 0.02957451, 0.021483,    0.01550561,
                       0.01097544, 0.0078939,  0.00526108, 0.00299911])

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
        'sampler': 'evaluate',
        'box': True,
        'z_snapshot': 0,
        'num_k': 10001,
        'rp_model_min': 7.391,
        'rp_model_max': 128.016,
        'bins_rp_model': 16,
        'log10kmin': -5,
        'log10kmax': 2,
        'l_min': 0,
        'l_max': 10001,
        'steps_l': 10,
        'H0': 68.1,
        'Om_m': 0.306,
        'Om_b': 0.0486,
        'sigma8': 0.807,
        'n_s': 0.967,
        'IA_model': 'TATT',
        'min_scale_cut': 5,
        'max_scale_cut': 100
    })
    
    run_module.config_setup = update_config(cfg)

    rp, wgg, wgp = run_module.run()

    assert wgg.shape == wgp.shape, "wgg and wgp shapes do not match"

    # Compare arrays using allclose
    assert np.allclose(rp, RP_GOLDEN, rtol=1e-3, atol=1e-8), "rp does not match golden reference"
    assert np.allclose(wgg, WGG_GOLDEN, rtol=1e-3, atol=1e-8), "wgg does not match golden reference"
    assert np.allclose(wgp, WGP_GOLDEN, rtol=1e-3, atol=1e-8), "wgp does not match golden reference"
