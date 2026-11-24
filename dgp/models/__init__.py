def build_model(config):
    """INR (shift-modulated SIREN), GINO or Transolver++ from the 'model' section of a config."""
    m = config['model']
    architecture = m['architecture']
    if architecture == "siren":
        from .siren import ShiftModulatedMLP
        return ShiftModulatedMLP(
            in_features=m['in_features_siren'],
            shift_size=m['shift_size_siren'],
            out_features=m['out_dim'],
            num_hidden_layers=m['num_hidden_layers_siren'],
            hidden_features=m['hidden_features_siren'],
            omega=float(m['omega_siren']),
            omega_0=float(m['omega_0_siren']),
            activation_fn=m['activation_fn_siren'],
            dropout_rate=m['dropout_rate_siren'],
            use_resnet=m['use_resnet_siren'],
            mod_activation_fn=m.get('mod_activation_fn_siren', 'identity'))
    if architecture == "gino":
        from .gino import UnifiedGINOWrapper
        return UnifiedGINOWrapper(config)
    if architecture == "transolver":
        from .transolver import UnifiedTransolverWrapper
        return UnifiedTransolverWrapper(config)
    raise ValueError(f"Unsupported architecture: {architecture}")
