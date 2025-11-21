# tests/conftest.py
import pytest
import numpy as np
import astropy.units as u
from IATheory import run as run_module

@pytest.fixture
def reset_config():
    """
    Provides a fresh copy of config_setup for each test.
    Patches the global config in run_module temporarily.
    """
    cfg = {
        "z_min": 0.0,  # Minimum redshift to model
        "z_max": 1.1,  # Maximum redshift to model
        "z_snapshot": 0,  # Redshift of the snapshot
        "num_k": 10001,
        "bins_z": 100,  # Number of redshift bins
        "rp_model_min": 7.391,  # Minimum transverse distance to model in Mpc
        "rp_model_max": 128.016,  # Maximum transverse distance to model in Mpc
        "bins_rp_model": 16,  # Number of transverse distance bins
        "log10kmin": -5,  # Minimum k
        "log10kmax": 2,   # Maximum k
        "l_min": 0,       # Minimum l
        "l_max": 10001,   # Maximum l
        "steps_l": 10,    # Steps in l
        "H0": 68.1,
        "Om_m": 0.306,    # Omega matter
        "Om_b": 0.0486,   # Omega baryons
        "sigma8": 0.807,
        "n_s": 0.967,
        "IA_model": "TATT",  # Model for intrinsic alignments
        "min_scale_cut": 5,  # Minimum scale cut in Mpc/h
        "max_scale_cut": 100,
        "z_type": "spec",  # Either "phot" or "spec"
        "Pi": np.array([-233, -144, -89, -55, -34, -21, -13, -8, -5, -3, -2, -1, 0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]) * u.Mpc / 0.69,
        "bins_zm": 100,  # Number of redshift bins for photometric errors
        "add_magnification": True,
        "add_galaxy_galaxy_lensing": True,
        "sampler": "evaluate",  # Choice between 'evaluate' and 'emcee'
        "box": True,  # Is it a box or a lightcone
        "n_cores": 192
    }
    orig_config = run_module.config_setup.copy()
    yield cfg
    run_module.config_setup = orig_config  # restore after test
