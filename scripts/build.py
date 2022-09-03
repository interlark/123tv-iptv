#!/usr/bin/env python3

import os
import pathlib
import shutil
import subprocess
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
CLEANUP = True


# Target filename
dist_filename = '123tv-iptv'

try:
    src_path = ROOT_DIR / 'main.py'

    # Dist
    build_cmd = [
        'pyinstaller',
        '--clean',
        '--onefile',
        '--windowed',
        '-n', dist_filename,
    ]

    # Icon
    if sys.platform.startswith('win'):
        build_cmd += [
            '--icon', ROOT_DIR / 'gui' / 'icons' / 'icon.ico',
        ]
    elif sys.platform.startswith('darwin'):
        build_cmd += [
            '--icon', ROOT_DIR / 'gui' / 'icons' / 'icon.icns',
        ]

    # Add data
    build_cmd += [
        '--add-data', f'{ROOT_DIR}/gui/icons{os.pathsep}gui/icons',
    ]

    # 123TV-IPTV fix
    iptv_module_dir = pathlib.Path(
        __import__('123tv_iptv').__file__
    ).parent

    # Furl fix
    orderedmultidict_version_file = pathlib.Path(
        __import__('orderedmultidict').__file__
    ).parent / '__version__.py'

    build_cmd += [
        '--add-data', f'{orderedmultidict_version_file}{os.pathsep}./orderedmultidict',
        '--add-data', f'{iptv_module_dir}{os.pathsep}{iptv_module_dir.name}',
        '--add-data', f'{iptv_module_dir}/channels.json{os.pathsep}.',
    ]

    # Source script
    build_cmd += [
        str(src_path),
    ]

    # Hidden import
    build_cmd += [
        '--hidden-import', 'tqdm.asyncio',
        '--hidden-import', 'aiohttp',
        '--hidden-import', 'netifaces',
        '--hidden-import', 'furl',
    ]

    cmd_line = ' '.join(str(arg) for arg in build_cmd)
    print('Running command: %s' % cmd_line, file=sys.stderr)
    subprocess.check_call(build_cmd)
finally:
    # Cleanup
    if CLEANUP:
        shutil.rmtree(ROOT_DIR / 'build', ignore_errors=True)
        try:
            os.remove(ROOT_DIR / f'{dist_filename}.spec')
        except FileNotFoundError:
            pass