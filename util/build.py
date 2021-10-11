def build_with_config(config, registry, default_args=None):
    '''build a model with config dict
       Parameters:
        config(dict): config dict to build the model, it should have key 'type'
        registry(Registry):
    '''
    if 'type' not in config:
        raise KeyError('The key "type" should be in config dict')
    args = config.copy()
    if default_args is not None:
        for k,v in default_args.items():
            args.setdefault(k,v)
    module_type = args.pop('type')
    module = registry.get(module_type)
    module = module(**args)
    return module
