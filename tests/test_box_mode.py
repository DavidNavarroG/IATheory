import pytest
import os
from IATheory import run as run_module
from IATheory.run import update_config

@pytest.mark.parametrize("box_option", [True, False])
def test_box_modes(reset_config, box_option):
    if not box_option and not os.path.exists('/nfs/pic.es/user/d/dnavarro/IATheory/data/nz/positions_nz.csv'):
        pytest.skip("Skipping box=False test: positions_nz.csv not available")
    
    cfg = reset_config
    cfg['sampler'] = 'evaluate'
    cfg['box'] = box_option
    run_module.config_setup = update_config(cfg)

    rp, wgg, wgp = run_module.run()

    assert wgg.shape == wgp.shape
    assert rp is not None or box_option is False
