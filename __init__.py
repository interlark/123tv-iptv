"""
123TV Free IPTV.
"""

import os
import importlib.util

import_list = ('main', 'args_parser', 'playlist_server', 'VERSION')

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, '123tv_iptv.py')
spec = importlib.util.spec_from_file_location('123tv_iptv', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

for attr in import_list:
    locals()[attr] = getattr(module, attr)

__version__ = module.VERSION
__all__ = import_list
