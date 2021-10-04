# adopted from https://github.com/open-mmlab/mmcv/blob/8cac7c25ee5bc199d6e4059297ef2fa92d9c069c/mmcv/utils/config.py
import os
import importlib
from addict import Dict

SUPPORT_EXT = ['.py']
RESERVED_KEYS = ['filename', 'text', 'pretty_text']
BASE_KEY = '_base_'
REPLACE_KEY = '_replace_'

class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

class Config:
    @staticmethod
    def _merge_dict_a_to_b(a, b):
        """merge dict ``a`` into dict ``b`` (non-inplace).
        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.
        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
        Returns:
            dict: The modified dict of ``b`` using ``a``.
        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}
            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}
            # b is a list
        """
        b = b.copy()
        for k, v in a.items():
            replace = v.pop(REPLACE_KEY, False)
            if isinstance(v,dict) and k in b and not replace:
                allowed_types = dict
                if not isinstance(b[k], allowed_types):
                    raise TypeError(
                        f'{k}={v} in child config cannot inherit from base '
                        f'because {k} is a dict in the child config but is of '
                        f'type {type(b[k])} in base config.' )
                b[k] = Config._merge_a_into_b(v, b[k])
            else:
                b[k] = v
        return b

    @staticmethod
    def _get_config_dict_from_file(file_path):
        # expand path to absolute path
        file_path = os.path.abspath(os.path.expanduser(file_path))

        # check path exist
        if not os.path.exists(file_path):
            print('file {} does not exist'.format(file_path))

        file_ext = os.path.splitext(file_path)[1]
        if file_ext not in SUPPORT_EXT:
            raise ValueError('Config file {} is not supported yet. Only {} are supported!'.format(file_path, SUPPORT_EXT))
        
        if file_ext == '.py':
            spec=importlib.util.spec_from_file_location("temp",file_path)

            # creates a new module based on spec
            mod = importlib.util.module_from_spec(spec)

            # executes the module in its own namespace
            # when a module is imported or reloaded.
            spec.loader.exec_module(mod)
            config_dict = {n:v for n,v in mod.__dict__.items() if not n.startswith('__')}
        else:
            print('something wrong!!!')

        cfg_text = file_path + '\n'
        with open(file_path, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in config_dict:
            base_path = config_dict.pop(BASE_KEY)
            base_path = base_path if isinstance(base_path, list) else [base_path]

            base_cfgs = []
            base_texts = []
            for base in base_path:
                cfg, text = Config._get_config_dict_from_file(base)
                base_cfgs.append(cfg)
                base_texts.append(text)

            base_config_dict = {}
            for cfg in base_cfgs:
                duplicate_keys = base_config_dict.keys() & cfg.keys()
                if len(duplicate_keys) > 0:
                    raise KeyError('Duplicate key is not allowed among bases. '
                                   f'Duplicate keys: {duplicate_keys}')
                base_config_dict.update(cfg)

            config_dict = Config._merge_dict_a_to_b(config_dict, base_config_dict)

            # merge cfg_text
            base_texts.append(cfg_text)
            cfg_text = '\n'.join(base_texts)

        return config_dict, cfg_text

    @staticmethod
    def fromfile(filename):
        cfg_dict, cfg_text = Config._get_config_dict_from_file(filename)
        #if import_custom_modules and cfg_dict.get('custom_imports', None):
        #    import_modules_from_strings(**cfg_dict['custom_imports'])
        return Config(cfg_dict, cfg_text=cfg_text, filename=filename)

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)
