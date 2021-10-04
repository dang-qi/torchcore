class Registry():
    '''register modual so the module name can map to the class'''
    def __init__(self, name):
        '''name(string): name of the Registry'''
        self._name = name
        self._module_dict = dict()

    def _register(self, name, module, force):
        if name is None:
            name = module.__name__
        if name in self._module_dict and not force:
            raise ValueError('The module name "{}" is registered. ' \
                              'Please try to set force=True if you want to overwrite'.format(name))
        self._module_dict[name] = module
        return module
        
    def register(self, module=None, name=None, force=False):
        '''Register the module to Registry with the name or module.__name__ 
            when name=None. Can also be used as a decorator when module is None
        Parameters:
            module(obj): the target module need to be registered,
            name(string): the name to register the module
            force(bool): force overwiter the module_dict with new obj'''
        if module is not None:
            return self._register(name, module, force)

        def register_dec(module):
            return self._register(name, module, force)

        return register_dec

    def get(self, key):
        if key not in self._module_dict:
            raise KeyError('Key "{}" is not registered! Make sure you have the correct name.'.format(key))
        return self._module_dict[key]

    def __repr__(self) -> str:
        return "Registry of {}:\nThe modules are: \n {}".format(self._name, self._module_dict) 

    def __iter__(self):
        return iter(self._module_dict.items())

    # pyre-fixme[4]: Attribute must be annotated.
    __str__ = __repr__
