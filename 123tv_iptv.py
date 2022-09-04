#!/usr/bin/env python3

import argparse
import asyncio
import functools
import io
import json
import logging
import pathlib
import re
import sys
import time
from enum import Enum
from typing import Any, Awaitable, Callable, List, Optional

import aiohttp
import netifaces
from aiohttp import web
from furl import furl
from tqdm.asyncio import tqdm

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

Channel = TypedDict('Channel', {'id': int, 'stream_id': str, 'tvguide_id': str,
                                'name': str, 'category': str, 'language': str,
                                'ustvgo_id': str, 'stream_url': furl})

# Fix for https://github.com/pyinstaller/pyinstaller/issues/1113
''.encode('idna')

# Usage:
# $ ./123tv_iptv.py
# $ ./123tv_iptv.py --icons-for-light-bg
# $ ./123tv_iptv.py --access-logs --port 1234
#
# Install / uninstall service (Linux only)
# $ sudo -E ./123tv_iptv.py --icons-for-light-bg install-service
# $ sudo -E ./123tv_iptv.py uninstall-service
# $ sudo -E env "PATH=$PATH" 123tv-iptv --port 1234 install-service
# $ sudo -E env "PATH=$PATH" 123tv-iptv uninstall-service
#
# Run:
# mpv http://127.0.0.1:6464/123tv.m3u8
# vlc http://127.0.0.1:6464


class AuthKeyRefreshPolicy(Enum):
    REFRESH_ONE_AT_A_TIME = 1  # Perform lazy auth key refresh (Lazy strategy)
    REFRESH_ALL_AT_ONCE = 2  # Refresh all auth keys at once (Eager strategy)


VERSION = '0.1.2'
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36')
HEADERS = {'User-Agent': USER_AGENT, 'Referer': 'http://azureedge.xyz/'}
AUTH_KEY_REFRESH_POLICY = AuthKeyRefreshPolicy.REFRESH_ALL_AT_ONCE


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def root_dir() -> pathlib.Path:
    """Root directory."""
    if hasattr(sys, '_MEIPASS'):
        return pathlib.Path(sys._MEIPASS)  # type: ignore
    else:
        return pathlib.Path(__file__).parent


def load_dict(filename: str) -> Any:
    """Load root dictionary."""
    filepath = root_dir() / filename
    with open(filepath, encoding='utf-8') as f:
        return json.load(f)


def local_ip_addresses() -> List[str]:
    """Finding all local IP addresses."""
    ip_addresses: List[str] = []
    interfaces = netifaces.interfaces()
    for i in interfaces:
        iface = netifaces.ifaddresses(i).get(netifaces.AF_INET, [])
        ip_addresses.extend(x['addr'] for x in iface)

    return ip_addresses


async def gather_with_concurrency(n: int, *tasks: Awaitable[Any],
                                  show_progress: bool = True,
                                  progress_title: Optional[str] = None) -> Any:
    """Gather tasks with concurrency."""
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    gather = functools.partial(tqdm.gather, desc=progress_title) if show_progress \
        else asyncio.gather
    return await gather(*[sem_task(x) for x in tasks])  # type: ignore


async def retrieve_stream_url(channel: Channel, max_retries: int = 5) -> Optional[Channel]:
    """Retrieve stream URL from web player with retries."""
    url = 'http://123tv.live/watch/' + channel['stream_id']
    timeout, max_timeout = 2, 10
    exceptions = (asyncio.TimeoutError, aiohttp.ClientConnectionError,
                  aiohttp.ClientResponseError, aiohttp.ServerDisconnectedError)

    while True:
        try:
            async with aiohttp.TCPConnector(ssl=False) as connector:
                async with aiohttp.ClientSession(raise_for_status=True, connector=connector) as session:
                    async with session.get(url=url, timeout=timeout) as response:
                        # Get channel iframe
                        resp_html = await response.text()
                        match = re.search(r'"(?P<iframe_url>https?://.*?\.m3u8\?embed=true)"', resp_html)
                        if not match:
                            return None

                        # Get channel playlist URL
                        headers = {**HEADERS, 'Referer': url}
                        embed_url = match.group('iframe_url')
                        async with session.get(url=embed_url, timeout=timeout, headers=headers) as response:
                            resp_html = await response.text()
                            match = re.search(r'\'(?P<stream_url>https?://.*?\.m3u8)\'', resp_html)
                            if not match:
                                return None

                            # Check if it's a valid playlist
                            stream_url = match.group('stream_url')
                            async with session.get(url=stream_url, timeout=timeout,
                                                   headers=HEADERS) as response:
                                resp_m3u8 = await response.text()
                                if resp_m3u8.startswith('#EXTM3U'):
                                    channel['stream_url'] = furl(stream_url)
                                    return channel

                        return None

        except Exception as e:
            is_exc_valid = any([isinstance(e, exc) for exc in exceptions])
            if not is_exc_valid:
                raise

            timeout = min(timeout + 1, max_timeout)
            max_retries -= 1
            if max_retries <= 0:
                logger.error('Failed to get url %s', url)
                return None


def render_playlist(channels: List[Channel], host: str, use_uncompressed_tvguide: bool) -> str:
    """Render master playlist."""
    with io.StringIO() as f:
        base_url = furl(netloc=host, scheme='http')
        tvg_compressed_ext = '' if use_uncompressed_tvguide else '.gz'
        tvg_url = base_url / f'tvguide.xml{tvg_compressed_ext}'

        f.write('#EXTM3U url-tvg="%s" refresh="1800"\n\n' % tvg_url)
        for channel in channels:
            if channel.get('stream_url'):
                tvg_logo = base_url / 'logos' / (channel['ustvgo_id'] + '.png')
                stream_url = base_url / (channel['stream_id'] + '.m3u8')

                f.write(('#EXTINF:-1 tvg-id="{0[ustvgo_id]}" tvg-logo="{1}" '
                         'group-title="{0[category]}",{0[name]}\n'.format(channel, tvg_logo)))
                f.write(f'{stream_url}\n\n')

        return f.getvalue()


async def collect_urls(channels: List[Channel], parallel: int) -> List[Channel]:
    """Collect channel stream URLs from 123tv.live web players."""
    logger.info('Extracting stream URLs from 123TV. Parallel requests: %d.', parallel)
    retrieve_tasks = [retrieve_stream_url(channel) for channel in channels]
    channels = await gather_with_concurrency(parallel, *retrieve_tasks,
                                             progress_title='Collect URLs')

    channels_ok = [x for x in channels if x]
    report_msg = 'Extracted %d channels out of %d.'
    logger.info(report_msg, len(channels_ok), len(channels))

    return channels_ok


async def playlist_server(port: int, parallel: bool, tvguide_base_url: str,
                          access_logs: bool, icons_for_light_bg: bool,
                          use_uncompressed_tvguide: bool) -> None:
    """Run proxying server with keys rotation."""
    async def master_handler(request: web.Request) -> web.Response:
        """Master playlist handler."""
        return web.Response(
            text=render_playlist(channels, request.host, use_uncompressed_tvguide)
        )

    async def logos_handler(request: web.Request) -> web.Response:
        """Channel logos handler."""
        color_scheme = 'for-light-bg' if icons_for_light_bg else 'for-dark-bg'
        logo_url = (furl(tvguide_base_url) / 'images/icons/channels' /
                    color_scheme / request.match_info.get('filename')).url

        async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
            async with aiohttp.request(method=request.method, url=logo_url,
                                       connector=connector) as response:
                content = await response.read()

                return web.Response(
                    body=content, status=response.status,
                    content_type='image/png'
                )

    async def keys_handler(request: web.Request) -> web.Response:
        """AES keys handler."""
        url = furl('http://hls.123tv.live/key/').add(path=request.match_info.get('keypath')).url
        async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
            async with aiohttp.request(
                method=request.method, url=url,
                headers=HEADERS, connector=connector
            ) as response:
                content = await response.read()

                return web.Response(
                    body=content, status=response.status
                )

    async def tvguide_handler(request: web.Request) -> web.Response:
        """TV Guide handler."""
        is_compressed = request.path.endswith('.gz')
        compressed_ext = '.gz' if is_compressed else ''
        color_scheme = 'for-light-bg' if icons_for_light_bg else 'for-dark-bg'
        tvguide_filename = f'ustvgo.{color_scheme}.xml{compressed_ext}'
        tvguide_url = furl(tvguide_base_url).add(path=tvguide_filename).url

        async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
            async with aiohttp.request(method=request.method, url=tvguide_url,
                                       connector=connector) as response:
                content = await response.read()
                content_type = 'application/gzip' if is_compressed else 'application/xml'

                return web.Response(
                    body=content, status=response.status,
                    content_type=content_type
                )

    async def playlist_handler(request: web.Request) -> web.Response:
        """Channel playlist handler."""
        stream_id = request.match_info.get('stream_id')
        if stream_id not in streams:
            return web.Response(text='Stream not found!', status=404)

        channel = streams[stream_id]
        headers = {name: value for name, value in request.headers.items()
                   if name not in (aiohttp.hdrs.HOST, aiohttp.hdrs.USER_AGENT)}
        headers = {**headers, **HEADERS}
        max_retries = 2  # Second retry for auth keys refreshing

        for _ in range(max_retries):
            async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
                async with aiohttp.request(
                    method=request.method, url=channel['stream_url'].url,
                    headers=headers, connector=connector
                ) as response:
                    # Proxy playlist's chunks
                    content = await response.text()
                    content = re.sub(r'(?<=\n)(https?)://', r'/chunks/\1/', content)

                    # Check if we need a key / the keys refresh
                    if response.status == 410:
                        if AUTH_KEY_REFRESH_POLICY == AuthKeyRefreshPolicy.REFRESH_ONE_AT_A_TIME:
                            # Lazy auth key update
                            async with channel['refresh_lock']:
                                if time.time() - channel['last_time_refreshed'] > 5:
                                    logger.info('Refreshing auth key for channel %s.', channel['name'])
                                    await retrieve_stream_url(channel, 2)
                                    channel['last_time_refreshed'] = time.time()

                        elif AUTH_KEY_REFRESH_POLICY == AuthKeyRefreshPolicy.REFRESH_ALL_AT_ONCE:
                            # Update all the channels keys at once
                            async with channels_refresh_lock:
                                nonlocal channels_last_time_refreshed  # type: ignore
                                if time.time() - channels_last_time_refreshed > 5:
                                    logger.info('Refreshing auth keys for all the channels.')
                                    await collect_urls(channels, parallel)
                                    channels_last_time_refreshed = time.time()

                        else:
                            raise ValueError('Unknown strategy for auth key refreshing.')

                        continue

                    # Not a valid playlist
                    if not content.startswith('#EXTM3U'):
                        logger.warning('%s returned invalid playlist: %s', response.url, content)
                        notfound_segment_url = furl(tvguide_base_url) / 'assets/404.ts'
                        return web.Response(text=(
                            '#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n'
                            f'#EXTINF:10.000\n{notfound_segment_url}\n#EXT-X-ENDLIST'
                        ))

                    # OK
                    return web.Response(text=content, status=200)

        # Failed to get new auth key / keys
        return web.Response(status=403)

    async def chunks_handler(request: web.Request) -> web.Response:
        """Chunks handler."""
        headers = {name: value for name, value in request.headers.items()
                   if name not in (aiohttp.hdrs.HOST, aiohttp.hdrs.USER_AGENT)}
        headers = {**headers, **HEADERS}
        url = '{0[schema]}://{0[chunk_url]}'.format(request.match_info)
        max_retries = 2  # Second retry for response payload errors recovering

        for retry in range(1, max_retries + 1):
            try:
                async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
                    async with aiohttp.request(
                        method=request.method, url=url, params=request.query,
                        headers=headers, raise_for_status=True, connector=connector
                    ) as response:

                        content = await response.read()
                        headers = {name: value for name, value in response.headers.items()
                                   if name not in
                                   (aiohttp.hdrs.CONTENT_ENCODING, aiohttp.hdrs.CONTENT_LENGTH,
                                    aiohttp.hdrs.TRANSFER_ENCODING, aiohttp.hdrs.CONNECTION)}

                        return web.Response(
                            body=content, status=response.status,
                            headers=headers
                        )

            except aiohttp.ClientResponseError as e:
                if retry >= max_retries:
                    return web.Response(text=e.message, status=e.status)

            except aiohttp.ClientPayloadError as e:
                if retry >= max_retries:
                    return web.Response(text=str(e), status=500)

            except aiohttp.ClientError as e:
                logger.error('[Retry %d/%d] Error occured during handling request: %s',
                             retry, max_retries, e, exc_info=True)
                if retry >= max_retries:
                    return web.Response(text=str(e), status=500)

        return web.Response(text='', status=500)

    # Load channels info
    channels = load_dict('channels.json')

    # Retrieve available channels with their stream urls
    channels = await collect_urls(channels, parallel)
    if not channels:
        logger.error('No channels were retrieved!')
        return

    # Add channels sync tools
    if AUTH_KEY_REFRESH_POLICY == AuthKeyRefreshPolicy.REFRESH_ONE_AT_A_TIME:
        # Lazy auth key refresh
        for channel in channels:
            channel['refresh_lock'] = asyncio.Lock()  # Lock on a key refresh
            channel['last_time_refreshed'] = .0  # Time of the last key refresh

    elif AUTH_KEY_REFRESH_POLICY == AuthKeyRefreshPolicy.REFRESH_ALL_AT_ONCE:
        # Refresh all the keys at once
        channels_refresh_lock = asyncio.Lock()  # Lock on the keys refresh
        channels_last_time_refreshed = .0  # Time of the last keys refresh

    else:
        raise ValueError('Unknown strategy for auth key refreshing.')

    # Transform list into a map for better accessibility
    streams = {x['stream_id']: x for x in channels}

    # Setup access logging
    access_logger = logging.getLogger('aiohttp.access')
    if access_logs:
        access_logger.setLevel('INFO')
    else:
        access_logger.setLevel('ERROR')

    # Run server
    for ip_address in local_ip_addresses():
        logger.info(f'Serving http://{ip_address}:{port}/123tv.m3u8')
        logger.info(f'Serving http://{ip_address}:{port}/tvguide.xml')

    app = web.Application()
    app.router.add_get('/', master_handler)  # master shortcut
    app.router.add_get('/123tv.m3u8', master_handler)  # master
    app.router.add_get('/tvguide.xml', tvguide_handler)  # tvguide
    app.router.add_get('/tvguide.xml.gz', tvguide_handler)  # tvguide compressed
    app.router.add_get('/logos/{filename:[^/]+}', logos_handler)  # logos
    app.router.add_get('/{stream_id}.m3u8', playlist_handler)  # playlist
    app.router.add_get('/chunks/{schema}/{chunk_url:.+}', chunks_handler)  # chunks
    app.router.add_get('/key/{keypath:.+}', keys_handler)  # AES

    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, port=port)
        await site.start()

        # Sleep forever by 1 hour intervals,
        # on Windows before Python 3.8 wake up every 1 second to handle
        # Ctrl+C smoothly.
        if sys.platform == 'win32' and sys.version_info < (3, 8):
            delay = 1
        else:
            delay = 3600

        while True:
            await asyncio.sleep(delay)

    finally:
        await runner.cleanup()  # Cleanup used resources, release port


def service_command_handler(command: str, *exec_args: str) -> bool:
    """Linux service command handler."""
    import os
    import subprocess
    import textwrap

    service_path = '/etc/systemd/system/123tv-iptv.service'
    service_name = os.path.basename(service_path)
    ret_failed = True

    def run_shell_commands(*commands: str) -> None:
        for command in commands:
            subprocess.run(command, shell=True)

    def install_service() -> bool:
        """Install systemd service."""
        service_content = textwrap.dedent(f'''
            [Unit]
            Description=123TV Free IPTV
            After=network.target
            StartLimitInterval=0

            [Service]
            User={os.getlogin()}
            Type=simple
            Restart=always
            RestartSec=5
            ExecStart={' '.join(exec_args)}

            [Install]
            WantedBy=multi-user.target
        ''')

        if os.path.isfile(service_path):
            logger.error('Service %s already exists!', service_path)
            return True

        with open(service_path, 'w') as f_srv:
            f_srv.write(service_content.strip())

        os.chmod(service_path, 0o644)

        run_shell_commands(
            'systemctl daemon-reload',
            'systemctl enable %s' % service_name,
            'systemctl start %s' % service_name
        )

        return False

    def uninstall_service() -> bool:
        """Uninstall systemd service."""
        if not os.path.isfile(service_path):
            logger.error('Service %s does not exist!', service_path)
            return True

        run_shell_commands(
            'systemctl stop %s' % service_name,
            'systemctl disable %s' % service_name
        )

        os.remove(service_path)

        run_shell_commands(
            'systemctl daemon-reload',
            'systemctl reset-failed'
        )

        return False

    try:
        if command == 'install-service':
            ret_failed = install_service()
        elif command == 'uninstall-service':
            ret_failed = uninstall_service()
        else:
            logger.error('Unknown command "%s"', command)

    except PermissionError:
        logger.error(('Permission denied, try command: '
                      f'sudo -E {" ".join(exec_args)} {command}'))
    except Exception as e:
        logger.error('Error occured: %s', e)

    return ret_failed


def args_parser() -> argparse.ArgumentParser:
    """Command line arguments parser."""
    def int_range(min_value: int = -sys.maxsize - 1,
                  max_value: int = sys.maxsize) -> Callable[[str], int]:
        def constrained_int(arg: str) -> int:
            value = int(arg)
            if not min_value <= value <= max_value:
                raise argparse.ArgumentTypeError(
                    f'{min_value} <= {arg} <= {max_value}'
                )
            return value

        return constrained_int

    parser = argparse.ArgumentParser(
        '123tv-iptv', description='123TV Free IPTV.', add_help=False
    )
    parser.add_argument(
        '-p', '--port', metavar='PORT',
        type=int_range(min_value=1, max_value=65535), default=6464,
        help='Serving port (default: %(default)s)'
    )
    parser.add_argument(
        '-t', '--parallel', metavar='N',
        type=int_range(min_value=1), default=10,
        help='Number of parallel parsing requests (default: %(default)s)'
    )
    parser.add_argument(
        '--icons-for-light-bg', action='store_true',
        help='Put channel icons adapted for apps with light background'
    )
    parser.add_argument(
        '--access-logs',
        action='store_true',
        help='Enable access logging'
    )
    parser.add_argument(
        '--tvguide-base-url', metavar='URL',
        default='https://raw.githubusercontent.com/interlark/ustvgo-tvguide/master',
        help='Base TV Guide URL'
    )
    parser.add_argument(
        '--use-uncompressed-tvguide',
        action='store_true',
        help='Use uncompressed version of TV Guide in "url-tvg" attribute'
    )
    parser.add_argument(
        '-v', '--version', action='version', version=f'%(prog)s {VERSION}',
        help='Show program\'s version number and exit'
    )
    parser.add_argument(
        '-h', '--help', action='help', default=argparse.SUPPRESS,
        help='Show this help message and exit'
    )

    # Linux service subcommands
    if sys.platform.startswith('linux'):
        subparsers = parser.add_subparsers(help='Subcommands')
        install_service_parser = subparsers.add_parser(
            'install-service', help='Install autostart service'
        )
        install_service_parser.set_defaults(
            invoke_subcommand=functools.partial(service_command_handler, 'install-service')
        )
        uninstall_service_parser = subparsers.add_parser(
            'uninstall-service', help='Uninstall autostart service'
        )
        uninstall_service_parser.set_defaults(
            invoke_subcommand=functools.partial(service_command_handler, 'uninstall-service')
        )

    return parser


def main() -> None:
    """Entry point."""
    # Parse CLI arguments
    parser = args_parser()
    args = parser.parse_args()

    # Invoke subcommands
    if 'invoke_subcommand' in args:
        exec_args = [arg for idx, arg in enumerate(sys.argv)
                     if arg.startswith('-') or idx == 0]
        exit(args.invoke_subcommand(*exec_args))

    # Run server
    try:
        asyncio.run(playlist_server(**vars(args)))
    except KeyboardInterrupt:
        logger.info('Server shutdown.')


if __name__ == '__main__':
    main()
