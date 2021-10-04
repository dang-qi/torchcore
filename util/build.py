def build_with_config(config, registry):
    '''build a model with config dict
       Parameters:
        config(dict): config dict to build the model, it should have key 'type'
        registry(Registry):
    '''
    if 'type' not in config:
        raise KeyError('The key "type" should be in config dict')
    args = config.copy()
    module_type = args.pop('type')
    module = registry.get(module_type)
    module = module(**args)
    return module
