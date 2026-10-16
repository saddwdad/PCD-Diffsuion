def build_model(config):
    args=dict(config)
    kind=args.pop('type','dense_reference')
    if kind=='dense_reference':
        from .model import PCDDiffusion
        return PCDDiffusion(**args)
    if kind=='prior_tokens':
        from .prior_token_model import PriorTokenDiffusion
        return PriorTokenDiffusion(**args)
    raise ValueError(f'Unknown model type: {kind}')
