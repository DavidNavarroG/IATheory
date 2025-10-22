config_setup = None  # Will be initialized by run.py

def init_config(cfg):
    """Initialize the global configuration."""
    global config_setup
    config_setup = cfg

def get_config():
    """Safely return the global configuration."""
    if config_setup is None:
        raise RuntimeError("Config not initialized! Call init_config() first.")
    return config_setup
