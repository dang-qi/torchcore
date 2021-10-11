# adopted from https://github.com/open-mmlab/mmcv/blob/8cac7c25ee5bc199d6e4059297ef2fa92d9c069c/mmcv/utils/config.py
from math import exp
import os
import importlib
from addict import Dict
import functools

from yapf.yapflib.yapf_api import FormatCode

SUPPORT_EXT = ['.py']
RESERVED_KEYS = ['filename', 'text', 'pretty_text', 'path_config']
BASE_KEY = '_base_'
REPLACE_KEY = '_replace_'

def mkdir( d ):
    if not os.path.isdir( d ) :
        os.mkdir( d )

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


# set attribute for nested config
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

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
            cfg_dir_path = os.path.dirname(file_path)
            base_path = config_dict.pop(BASE_KEY)
            base_path = base_path if isinstance(base_path, list) else [base_path]

            base_cfgs = []
            base_texts = []
            for base in base_path:
                cfg, text = Config._get_config_dict_from_file(os.path.join(cfg_dir_path,base))
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

    def initialize_project( self, project_name, project_base_path, config_name=None, tag=None ):
        self._cfg_dict.project_name = project_name
        self._cfg_dict.project_base_path = project_base_path
        path_config=ConfigDict()
        project_path = os.path.join(project_base_path, project_name)
        mkdir(project_path)
        path_config.project_path = project_path

        if config_name is None:
            # infer config name from config file name
            assert self._filename is not None
            config_name = os.path.splitext(os.path.basename(self._filename))[0]
        
        if tag is not None:
            config_name = config_name+'_'+tag
        else:
            tag=''

        self._cfg_dict.config_name = config_name

        exp_dir = os.path.join(project_path, config_name)
        path_config.exp_dir=exp_dir

        mkdir(exp_dir)
        log_path = os.path.join(exp_dir,'logs')
        checkpoint_path = os.path.join(exp_dir,'checkpoints')
        config_path = os.path.join(exp_dir, tag+self.filename)
        mkdir(log_path)
        mkdir(checkpoint_path)
        path_config.config_path = config_path
        path_config.log_dir = log_path
        path_config.log_path = os.path.join(log_path, tag+'.log')
        path_config.checkpoint_path = checkpoint_path
        path_config.checkpoint_path_tmp = '{}/checkpoint_{}_{{}}.pth'.format(checkpoint_path, tag)
        self._cfg_dict.path_config=path_config

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
    def path_config(self):
        return self._cfg_dict.path_config

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

    def merge_args(self, args, mapping_dict=None):
        '''merge the predifined args from args'''
        default_mapping = dict(batch_size='dataloader_train.batch_size',
                            accumulation_step='trainer.accumulation_step')
        if mapping_dict is None:
            mapping_dict = default_mapping

        for k_arg, k_cfg in mapping_dict.items():
            rsetattr(self._cfg_dict, k_cfg, getattr(args,k_arg))


    def dump(self, file=None):
        #cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if self.filename.endswith('.py'):
            if file is None:
                return self.pretty_text
            else:
                with open(file, 'w') as f:
                    f.write(self.pretty_text)
        else:
            raise NotImplementedError('It just support .py format now')

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        text, _ = FormatCode(text, style_config=yapf_style, verify=True)

        return text

    