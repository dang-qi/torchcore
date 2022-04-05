def build_with_config(config, registry, default_args=None, non_copy_key=[]):
    '''build a model with config dict
       Parameters:
        config(dict): config dict to build the model, it should have key 'type'
        registry(Registry):
        non_copy_key: there are some object should be pass to args without copy
    '''
    if 'type' not in config:
        raise KeyError('The key "type" should be in config dict')
    args = config.copy()
    for k in non_copy_key:
        args[k] = config[k]
    if default_args is not None:
        for k,v in default_args.items():
            args.setdefault(k,v)
    module_type = args.pop('type')
    module = registry.get(module_type)
    module = module(**args)
    return module
