#!/usr/bin/env python3

import argparse
import asyncio
import base64
import binascii
import functools
import hashlib
import io
import json
import logging
import pathlib
import re
import sys
import textwrap
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import aiohttp
import netifaces
import pyaes
from aiohttp import web
from furl import furl
from pyaes.util import strip_PKCS7_padding
from tqdm.asyncio import tqdm

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

Channel = TypedDict('Channel', {'id': int, 'stream_id': str, 'tvguide_id': str,
                                'name': str, 'category': str, 'language': str,
                                'stream_url': furl, 'referer_url': str,
                                'refresh_lock': asyncio.Lock, 'last_time_refreshed': float})

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


VERSION = '0.1.3'
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/102.0.5005.63 Safari/537.36')
HEADERS = {'User-Agent': USER_AGENT}


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


def extract_obfuscated_link(content: str) -> str:
    """Extract and decode obfuscated stream URL from 123TV channel page."""
    start_text = textwrap.dedent(
        """
            var post_id = parseInt($('#'+'v'+'i'+'d'+'eo'+'-i'+'d').val());
            $(document).ready(function(){
        """
    )

    data_part, key = '', ''
    ob_line = content[content.index(start_text) + len(start_text):].splitlines()[0]
    try:
        a, b, c = ob_line.split('};')
    except ValueError:
        return ''

    # Parse data
    arr = re.search(r'(?<=\[)[^\]]+(?=\])', a)
    if arr:
        for part in arr.group().split(','):
            data_part += part.strip('\'')

    try:
        data = json.loads(base64.b64decode(data_part))
    except (binascii.Error, json.JSONDecodeError):
        return ''

    # Parse key
    for arr in re.findall(r'(?<=\[)[\d+\,]+(?=\])', b):
        for dig in arr.split(','):  # type: ignore
            key = chr(int(dig)) + key

    # Decode playlist data
    data['iterations'] = 999 if data['iterations'] <= 0 else data['iterations']
    dec_key = hashlib.pbkdf2_hmac('sha512', key.encode('utf8'), bytes.fromhex(data['salt']),
                                  data['iterations'], dklen=256 // 8)

    aes = pyaes.AESModeOfOperationCBC(dec_key, iv=bytes.fromhex(data['iv']))
    ciphertext = base64.b64decode(data['ciphertext'])
    decrypted = b''
    for idx in range(0, len(ciphertext), 16):
        decrypted += aes.decrypt(ciphertext[idx: idx + 16])

    target_link: str = strip_PKCS7_padding(decrypted).decode('utf8')
    path = re.search(r'(?<=\')\S+(?=\';}$)', c)
    if path:
        target_link += path.group()
    return target_link


def ensure_absolute_url(url: str, relative_url: str) -> str:
    """Ensure url is absolute. Makes it absolute if it's relative."""
    if not url.startswith('http'):
        url = furl(relative_url).origin + '/' + \
            '/'.join(furl(relative_url).path.segments[:-1] + [url])

    return url


async def retrieve_stream_url(channel: Channel, max_retries: int = 5) -> Optional[Channel]:
    """Retrieve stream URL from web player with retries."""
    url = 'http://123tv.live/watch/' + channel['stream_id']
    timeout, max_timeout = 2, 10
    exceptions = (asyncio.TimeoutError, aiohttp.ClientConnectionError,
                  aiohttp.ClientResponseError, aiohttp.ServerDisconnectedError)

    async def is_valid_playlist_url(url: str, headers: Dict[str, str],
                                    session: aiohttp.ClientSession) -> bool:
        async with session.get(url=url, timeout=timeout,
                               headers=headers) as response:
            m3u8_content = await response.text()
            return m3u8_content.startswith('#EXTM3U')  # type: ignore

    async def retrieve_regular_channel(html_content: str, session: aiohttp.ClientSession) -> bool:
        iframe_match = re.search(r'"(?P<iframe_url>https?://.*?\.m3u8\?embed=true)"', html_content)
        if not iframe_match:
            return False

        # Get channel playlist URL
        headers = {**HEADERS, 'Referer': url}
        embed_url = iframe_match.group('iframe_url')

        async with session.get(url=embed_url, timeout=timeout, headers=headers) as response:
            html_frame = await response.text()
            playlist_match = re.search(r'\'(?P<playlist_url>https?://.*?\.m3u8)\'', html_frame)
            if not playlist_match:
                return False

            # Check if it's a valid playlist
            playlist_url = playlist_match.group('playlist_url')
            referer_url = 'http://azureedge.xyz/'
            headers = {**HEADERS, 'Referer': referer_url}

            if await is_valid_playlist_url(playlist_url, headers, session):
                channel['stream_url'] = furl(playlist_url)
                channel['referer_url'] = referer_url
                return True

        return False

    async def retrieve_obfuscated_channel(html_content: str, session: aiohttp.ClientSession) -> bool:
        decoded_url = extract_obfuscated_link(html_content)
        if not decoded_url:
            return False

        referer_url = url.rstrip('/') + '/'
        headers = {**HEADERS, 'Referer': referer_url}

        async with session.get(url=decoded_url, timeout=timeout, headers=headers) as response:
            master_playlist_obj = await response.json()
            master_playlist_url = master_playlist_obj[0]['file']

            async with session.get(url=master_playlist_url, timeout=timeout, headers=headers) as response:
                master_playlist_content = await response.text()

                playlist_match = re.search(r'(?P<playlist_url>^[^#].*\.m3u8.*$)',
                                           master_playlist_content, re.M)
                if playlist_match:
                    playlist_url = ensure_absolute_url(playlist_match.group('playlist_url'),
                                                       master_playlist_url)

                    if await is_valid_playlist_url(playlist_url, headers, session):
                        channel['stream_url'] = furl(playlist_url)
                        channel['referer_url'] = referer_url
                        return True

        return False

    for key in ('stream_url', 'referer_url'):
        channel.pop(key, None)  # type: ignore

    while True:
        try:
            async with aiohttp.TCPConnector(ssl=False) as connector:
                async with aiohttp.ClientSession(raise_for_status=True, connector=connector) as session:
                    async with session.get(url=url, timeout=timeout) as response:
                        html_content = await response.text()

                        for retriever in (retrieve_regular_channel, retrieve_obfuscated_channel):
                            if await retriever(html_content, session):
                                return channel

                        logger.info('No stream URL found for channel "%s".', channel['name'])
                        return None

        except Exception as e:
            is_exc_valid = any(isinstance(e, exc) for exc in exceptions)
            if not is_exc_valid:
                raise

            timeout = min(timeout + 1, max_timeout)
            max_retries -= 1
            if max_retries <= 0:
                logger.debug('Failed to retrieve channel "%s" (%s).', channel['name'], url)
                return None


def render_playlist(channels: List[Channel], host: str, use_uncompressed_tvguide: bool) -> str:
    """Render master playlist."""
    with io.StringIO() as f:
        base_url = furl(netloc=host, scheme='http')
        tvg_compressed_ext = '' if use_uncompressed_tvguide else '.gz'
        tvg_url = base_url / f'tvguide.xml{tvg_compressed_ext}'

        f.write('#EXTM3U url-tvg="%s" refresh="3600"\n\n' % tvg_url)
        for channel in channels:
            tvg_logo = base_url / 'logos' / (channel['stream_id'] + '.png')
            stream_url = base_url / (channel['stream_id'] + '.m3u8')

            f.write(('#EXTINF:-1 tvg-id="{0[stream_id]}" tvg-logo="{1}" '
                     'group-title="{0[category]}",{0[name]}\n'.format(channel, tvg_logo)))
            f.write(f'{stream_url}\n\n')

        return f.getvalue()


async def collect_urls(channels: List[Channel], parallel: int,
                       keep_all_channels: bool) -> List[Channel]:
    """Collect channel stream URLs from 123tv.live web players."""
    logger.info('Extracting stream URLs from 123TV. Parallel requests: %d.', parallel)
    retrieve_tasks = [retrieve_stream_url(channel) for channel in channels]
    retrieved_channels = await gather_with_concurrency(parallel, *retrieve_tasks,
                                                       progress_title='Collect URLs')

    channels_ok = channels if keep_all_channels else \
        [x for x in retrieved_channels if x]

    report_msg = 'Extracted %d channels out of %d.'
    logger.info(report_msg, len(channels_ok), len(channels))

    return channels_ok


def preprocess_playlist(content: str, referer_url: str, response_url: Optional[str] = None) -> str:
    """Augment playlist with referer argument, make relative URLs absolute
    if `response_url` is specified, add chunks prefix path for absolute URLs."""
    if content.startswith('#EXTM3U'):
        content_lines = []
        for line in content.splitlines():
            if not line.startswith('#'):
                # Ensure URL is absolute
                if response_url:
                    line = ensure_absolute_url(line, response_url)

                # Add referer argument
                line = furl(line).add(args={'referer': referer_url}).url

            content_lines.append(line)

        content = '\n'.join(content_lines)

        # Add chunks redirect prefix path
        content = re.sub(r'(?<=\n)(https?)://', r'/chunks/\1/', content)

    return content


async def playlist_server(port: int, parallel: bool, tvguide_base_url: str,
                          access_logs: bool, icons_for_light_bg: bool,
                          use_uncompressed_tvguide: bool,
                          keep_all_channels: bool) -> None:
    async def refresh_auth_key(channel: Channel) -> None:
        """Refresh auth key for the channel."""
        async with channel['refresh_lock']:
            if time.time() - channel['last_time_refreshed'] > 5:
                logger.info('Refreshing auth key for channel %s.', channel['name'])
                await retrieve_stream_url(channel, 2)
                channel['last_time_refreshed'] = time.time()

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
            headers = {**HEADERS, 'Referer': 'http://azureedge.xyz/'}
            async with aiohttp.request(
                method=request.method, url=url,
                headers=headers, connector=connector
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
        tvguide_filename = f'123tv.{color_scheme}.xml{compressed_ext}'
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

        # If you specified --do-not-filter-channels
        # you could meet a channel without stream_url
        if 'stream_url' not in channel:
            # Try to refresh empty channel
            await refresh_auth_key(channel)

        if 'stream_url' in channel:
            headers = {name: value for name, value in request.headers.items()
                       if name not in (aiohttp.hdrs.HOST, aiohttp.hdrs.USER_AGENT)}
            headers = {**headers, **HEADERS, 'Referer': channel['referer_url']}

            max_retries = 2  # Second retry for auth keys refreshing
            for _ in range(max_retries):
                async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
                    async with aiohttp.request(
                        method=request.method, url=channel['stream_url'].url,
                        headers=headers, connector=connector
                    ) as response:
                        # Get playlist content
                        content = await response.text()

                        # Check if the channel's key is expired
                        if not content.startswith('#EXTM3U'):
                            # Lazy auth key update
                            await refresh_auth_key(channel)
                            continue

                        # Preprocess playlist content
                        content = preprocess_playlist(content, channel['referer_url'],
                                                      channel['stream_url'].url)

                        # OK
                        return web.Response(text=content, status=200)

        # Empty channel, not a valid playlist, failed to get new auth key
        logger.warning('Channel "%s" returned invalid playlist.', channel['name'])
        notfound_segment_url = furl(tvguide_base_url) / 'assets/404.ts'
        return web.Response(text=(
            '#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n'
            f'#EXTINF:10.000\n{notfound_segment_url}\n#EXT-X-ENDLIST'
        ))

    async def chunks_handler(request: web.Request) -> web.Response:
        """Chunks handler."""
        upstream_url = '{0[schema]}://{0[chunk_url]}'.format(request.match_info)
        upstream_query = {**request.query}
        referer_url = upstream_query.pop('referer', 'http://123tv.live/')

        headers = {name: value for name, value in request.headers.items()
                   if name not in (aiohttp.hdrs.HOST, aiohttp.hdrs.USER_AGENT)}
        headers = {**headers, **HEADERS, 'Referer': referer_url}

        max_retries = 2  # Second retry for response payload errors recovering
        for retry in range(1, max_retries + 1):
            try:
                async with aiohttp.TCPConnector(ssl=False, force_close=True) as connector:
                    async with aiohttp.request(
                        method=request.method, url=upstream_url, params=upstream_query,
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
    channels = await collect_urls(channels, parallel, keep_all_channels)
    if not channels:
        logger.error('No channels were retrieved!')
        return

    # Add channels sync tools
    for channel in channels:
        channel['refresh_lock'] = asyncio.Lock()  # Lock on a key refresh
        channel['last_time_refreshed'] = .0  # Time of the last key refresh

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
        type=int_range(min_value=1), default=15,
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
        '--keep-all-channels',
        action='store_true',
        help='Do not filter out not working channels'
    ),
    parser.add_argument(
        '--tvguide-base-url', metavar='URL',
        default='https://raw.githubusercontent.com/interlark/123tv-tvguide/master',
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
