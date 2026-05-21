"""
Tests for the evaluation mode of IATheory.

Run by using pytest -v -W ignore::DeprecationWarning in the root directory


This module validates that:
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
WGG_GOLDEN = np.array([2.14318078e+01,  1.85719272e+01,  1.57528884e+01,  1.32047721e+01,
                       1.07676917e+01,  8.67057520e+00,  6.73952874e+00,  5.10908373e+00,
                       3.72293868e+00,  2.61466757e+00,  1.72868103e+00,  1.07148694e+00,
                       5.83953389e-01,  2.44110663e-01,  1.10563605e-02, -1.31803771e-01])
WGP_GOLDEN = np.array([0.13463291, 0.10948468, 0.08806324, 0.06873266, 0.05330094,
                       0.0394877 , 0.02915717, 0.02075686, 0.01454301, 0.00975001,
                       0.00644277, 0.00403262, 0.00252759, 0.00151629, 0.00092489,
                       0.0004267])

def test_golden_values():
    """

    Verify that IATheory's evaluation mode matches the golden reference output.

    This test:
    - Sets up a known cosmology, IA model, and modeling scale configuration
    - Ensures wgg and wgp have identical shapes
    - Compares rp, wgg, and wgp against pre-computed golden values using `allclose`

    The golden arrays encode the expected results for this fixed configuration.
    Any deviation indicates a regression in physics modeling, numerical accuracy,
    or configuration handling.

    """
    cfg = ({
        'H0': 68.1,
        'Om_m': 0.306,
        'Om_b': 0.0486,
        'sigma8': 0.807,
        'n_s': 0.967,
        'num_k': 10001,
        'rp_model_min': 7.391,
        'rp_model_max': 128.016,
        'bins_rp_model': 16,
        'log10kmin': -5,
        'log10kmax': 2,
        'l_min': 0,
        'l_max': 10001,
        'steps_l': 10,
        'IA_model': 'TATT'
        })

    cfg = update_global_config(cfg)

    galaxy_bias = [1.2, -0.4]
    ia_params = [0.5, 1, 1.5]
    model = model_2p_corr(cfg, galaxy_bias, ia_params)

    case = 'lightcone_phot'
    config_setup_lightcone_phot = dict(z_min = 0., z_max = 1.1, z_type = 'phot',
                                       path_nz_positions = '/nfs/pic.es/user/d/dnavarro/IATheory/data/nz/positions_nz.csv',
                                       path_nz_shapes = '/nfs/pic.es/user/d/dnavarro/IATheory/data/nz/shapes_nz.csv',
                                       Pi = np.array([-233,-144,-89,-55,-34,-21,-13,-8,-5,-3,-2,-1,0,1,2,3,5,8,13,21,34,55,89,144,233]),
                                       bins_zm = 10, add_magnification = True, alpha_magnification = 0.93, add_galaxy_galaxy_lensing = True)
    config_lightcone_phot = build_specific_config(cfg, config_setup_lightcone_phot, case)

    model.model_wgg_phot_lightcone(config_lightcone_phot)
    model.model_wgp_phot_lightcone(config_lightcone_phot)

    assert model.wgg_phot_lightcone.xi.shape == model.wgp_phot_lightcone.xi.shape, "wgg and wgp shapes do not match"

    # Compare arrays using allclose
    assert np.allclose(cfg['rp_model'], RP_GOLDEN, rtol=1e-3, atol=1e-8), "rp does not match golden reference"
    assert np.allclose(model.wgg_phot_lightcone.xi, WGG_GOLDEN, rtol=1e-3, atol=1e-8), "wgg does not match golden reference"
    assert np.allclose(model.wgp_phot_lightcone.xi, WGP_GOLDEN, rtol=1e-3, atol=1e-8), "wgp does not match golden reference"
