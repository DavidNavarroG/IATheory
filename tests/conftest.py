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
        "H0": 68.1,
        "Om_m": 0.306,    # Omega matter
        "Om_b": 0.0486,   # Omega baryons
        "sigma8": 0.807,
        "n_s": 0.967,
        "num_k": 10001,
        "rp_model_min": 7.391,  # Minimum transverse distance to model in Mpc
        "rp_model_max": 128.016,  # Maximum transverse distance to model in Mpc
        "bins_rp_model": 16,  # Number of transverse distance bins
        "log10kmin": -5,  # Minimum k
        "log10kmax": 2,   # Maximum k
        "l_min": 0,       # Minimum l
        "l_max": 10001,   # Maximum l
        "steps_l": 10,    # Steps in l
        "IA_model": "TATT",  # Model for intrinsic alignments
        "min_scale_cut": 5,  # Minimum scale cut in Mpc/h
        "max_scale_cut": 100,
    }
    orig_config = run_module.config_setup.copy()
    yield cfg
    run_module.config_setup = orig_config  # restore after test
