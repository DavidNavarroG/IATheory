# tests/test_output.py
from IATheory import run as run_module
from IATheory.run import update_config


def test_print_outputs(reset_config, capsys):
    cfg = reset_config
    cfg['sampler'] = 'evaluate'
    run_module.config_setup = update_config(cfg)

    run_module.run()
    captured = capsys.readouterr()
    assert 'wgg' in captured.out
    assert 'wgp' in captured.out
