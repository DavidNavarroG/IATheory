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
from astropy import units as u
from IATheory import run as run_module
from IATheory.run import update_config

# Golden reference arrays
RP_GOLDEN = np.array([  7.391     ,   8.9386913 ,  10.81047248,  13.07420868,
                        15.8119761 ,  19.1230379 ,  23.12744318,  27.97037953,
                        33.82743717,  40.91097528,  49.4778215 ,  59.83858375,
                        72.36891191,  87.52311773, 105.85064686, 128.016])
WGG_GOLDEN = np.array([2.14318078e+01,  1.85719272e+01,  1.57528884e+01,  1.32047721e+01,
                       1.07676917e+01,  8.67057520e+00,  6.73952874e+00,  5.10908373e+00,
                       3.72293868e+00,  2.61466757e+00,  1.72868103e+00,  1.07148694e+00,
                       5.83953389e-01,  2.44110663e-01,  1.10563605e-02, -1.31803771e-01])
WGP_GOLDEN = np.array([0.13463291, 0.10948468, 0.08806324, 0.06873266, 0.05330094,
                       0.0394877 , 0.02915717, 0.02075686, 0.01454301, 0.00975001,
                       0.00644277, 0.00403262, 0.00252759, 0.00151629, 0.00092489,
                       0.0004267])

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
        'box': False,
        'z_min': 0,
        'z_max': 1.1,
        'bins_z':100,
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
        'max_scale_cut': 100,
        'z_type': 'phot',
        'Pi' : np.array([-233,-144,-89,-55,-34,-21,-13,-8,-5,-3,-2,-1,0,1,2,3,5,8,13,21,34,55,89,144,233])* u.Mpc,
        'bins_zm': 10,
        'add_magnification' : True,
        'add_galaxy_galaxy_lensing' : True
        })
    
    run_module.config_setup = update_config(cfg)

    rp, wgg, wgp = run_module.run()

    assert wgg.shape == wgp.shape, "wgg and wgp shapes do not match"

    # Compare arrays using allclose
    assert np.allclose(rp, RP_GOLDEN, rtol=1e-3, atol=1e-8), "rp does not match golden reference"
    assert np.allclose(wgg, WGG_GOLDEN, rtol=1e-3, atol=1e-8), "wgg does not match golden reference"
    assert np.allclose(wgp, WGP_GOLDEN, rtol=1e-3, atol=1e-8), "wgp does not match golden reference"
