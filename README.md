<div align="center">
    <h1>
        <a href="#">
            <img alt="123TV-IPTV Logo" width="50%" src="https://user-images.githubusercontent.com/20641837/188281506-11413220-3c65-4e26-a0d1-ed3f248ea564.svg"/>
        </a>
    </h1>
</div>


<div align="center">
    <a href="https://github.com/interlark/ustvgo-tvguide/actions/workflows/tvguide.yml"><img alt="TV Guide status" src="https://github.com/interlark/ustvgo-tvguide/actions/workflows/tvguide.yml/badge.svg"/></a>
    <a href="https://pypi.org/project/123tv-iptv"><img alt="PyPi version" src="https://badgen.net/pypi/v/123tv-iptv"/></a>
    <a href="https://pypi.org/project/123tv-iptv"><img alt="Supported platforms" src="https://badgen.net/badge/platform/Linux,macOS,Windows?list=|"/></a>
</div><br>

**123TV-IPTV** is an app that allows you to watch **free IPTV**.

It **extracts stream URLs** from [123tv.live](http://123tv.live/) website, **generates master playlist** with available TV channels for IPTV players and **proxies the traffic** between your IPTV players and streaming backends.

> **Note**: This is a port of [ustvgo-iptv app](https://github.com/interlark/ustvgo-iptv) for 123TV service.

## ✨ Features
- 🔑 Auto auth-key rotation
  > As server proxies the traffic it can detect if your auth key is expired and refresh it on the fly.
- 📺 Available [TV Guide](https://github.com/interlark/ustvgo-tvguide)
  > [TV Guide](https://github.com/interlark/ustvgo-tvguide) repo generates EPG XML for upcoming programs of all the channels twice an hour.
- [![](https://user-images.githubusercontent.com/20641837/173175879-aed31bd4-b188-4681-89df-5ffc3ea05a82.svg)](https://github.com/interlark/ustvgo-tvguide/tree/master/images/icons/channels)
Two iconsets for IPTV players with light and dark backgrounds
  > There are 2 channel iconsets adapted for apps with light and dark UI themes.
- 🗔 Cross-platform GUI
  > GUI is available for Windows, Linux and MacOS for people who are not that much into CLI.


## 🚀 Installation
- **CLI**
  ```bash
  pip install 123tv-iptv
  ```
- **GUI**

  You can download GUI app from [Releases](https://github.com/interlark/123tv-iptv/releases/latest) for your OS.
- **Docker**
  ```bash
  docker run -d --name=123tv-iptv -p 6464:6464 --restart unless-stopped ghcr.io/interlark/123tv-iptv:latest
  ```
  > For dark icons append following argument: `--icons-for-light-bg`

## ⚙️ Usage - CLI

  You can run the app without any arguments.
  ```
  123tv-iptv
  ```

  <img alt="123TV-IPTV CLI screencast" width="666" src="https://user-images.githubusercontent.com/20641837/188278700-4d3a6082-e955-4fe9-acba-00de48581b90.png"/>


| Optional argument                  | Description |
| :---                      |    :----   |
| --icons-for-light-bg      | Switch to dark iconset for players with light UI.       |
| --access-logs             | Enable access logs for tracking requests activity.        |
| --port 6464               | Server port. By default, the port is **6464**.           |
| --parallel 10             | Number of parallel parsing requests. Default is **10**.          |
| --use-uncompressed-tvguide| By default, master playlist has a link to **compressed** version of TV Guide:<br/>`url-tvg="http://127.0.0.1:6464/tvguide.xml.gz"`<br/>With this argument you can switch it to uncompressed:<br/>`url-tvg="http://127.0.0.1:6464/tvguide.xml"`           |

<br />

**Linux** users can install **systemd service** that automatically runs 123tv-iptv on start-ups ⏰.

```bash
# Install "123tv-iptv" service
sudo -E env "PATH=$PATH" 123tv-iptv install-service

# You can specify any optional arguments you want
sudo -E env "PATH=$PATH" 123tv-iptv --port 1234 --access-logs install-service

# Uninstall "123tv-iptv" service
sudo -E env "PATH=$PATH" 123tv-iptv uninstall-service
```

## ⚙️ Usage - GUI

  <img alt="123TV-IPTV GUI screenshot" width="614" src="https://user-images.githubusercontent.com/20641837/188281122-9b4ce62b-5088-4366-adbd-b4d421b3d302.png"/>

  If you don't like command line stuff, you can run GUI app and click "Start", simple as that.
  
  GUI uses **config file** on following path:

  * **Linux**: ~/.config/123tv-iptv/settings.cfg
  * **Mac**: ~/Library/Application Support/123tv-iptv/settings.cfg
  * **Windows**: C:\Users\\%USERPROFILE%\AppData\Local\123tv-iptv\settings.cfg

## 🔗 URLs
To play and enjoy your free IPTV you need 2 URLs that this app provides:
1) Your generated **master playlist**: 🔗 http://127.0.0.1:6464/123tv.m3u8
2) **TV Guide** (content updates twice an hour): 🔗 http://127.0.0.1:6464/tvguide.xml

## ▶️ Players
  Here is a **list** of popular IPTV players.
  
  **123TV**'s channels have **EIA-608** embedded subtitles. In case if you're not a native speaker and use *TV*, *Cartoons*, *Movies* and *Shows* to learn English and Spanish languages I would recommend you following free open-source cross-platform IPTV players that can handle EIA-608 subtitles:
  - **[VLC](https://github.com/videolan/vlc)**
  
      This old beast could play **any subtitles**. Unfortunately it **doesn't support TV Guide**.
      
      - **Play**
        ```bash
        vlc http://127.0.0.1:6464/123tv.m3u8
        ```
  - **[MPV](https://github.com/mpv-player/mpv)**
      
      Fast and extensible player. It **supports subtitles**, but not that good as VLC, sometimes you could encounter troubles playing roll-up subtitles. Unfortunately it **doesn't suppport TV Guide**.
      
      - **Play**
        ```bash
        mpv http://127.0.0.1:6464/123tv.m3u8
        ```
  - **[Jellyfin Media Player](https://github.com/jellyfin/jellyfin-media-player)**
    
    <img alt="Jellyfin Media Player screenshot" width="49%" src="https://user-images.githubusercontent.com/20641837/173175969-cbfe5adc-1dc8-4e3b-946c-fa4e295d8b8c.jpg"/>
    <img alt="Jellyfin Media Player screenshot" width="49%" src="https://user-images.githubusercontent.com/20641837/173175973-8acb076c-e1ac-4d06-96a8-b10a72b2f7d7.jpg"/>

    Comfortable, handy, extensible with smooth UI player. **Supports TV Guide**, has **mpv** as a backend.
    
    **Supports subtitles**, but there is no option to enable them via user interface. If you want to enable IPTV subtitles you have to use following "Mute" hack.
  
    - **Enable IPTV subtitles**
    
      I found a quick hack to force play embedded IPTV subtitles, all you need is to create one file:
    
      > Linux: `~/.local/share/jellyfinmediaplayer/scripts/subtitles.lua`
    
      > Linux(Flatpak): `~/.var/app/com.github.iwalton3.jellyfin-media-player/data/jellyfinmediaplayer/scripts/subtitles.lua`
    
      > MacOS: `~/Library/Application Support/Jellyfin Media Player/scripts/subtitles.lua`
    
      > Windows: `%LOCALAPPDATA%\JellyfinMediaPlayer\scripts\subtitles.lua`
    
      And paste following text in there:
    
      ```lua
      -- File: subtitles.lua
      function on_mute_change(name, value)
          if value then
              local subs_id = mp.get_property("sid")
              if subs_id == "1" then
                  mp.osd_message("Subtitles off")
                  mp.set_property("sid", "0")
              else
                  mp.osd_message("Subtitles on")
                  mp.set_property("sid", "1")
              end
          end
      end

      mp.observe_property("mute", "bool", on_mute_change)
      ```
      After that every time you mute a video *(🅼 key pressed)*, you toggle subtitles on/off as a side effect.

    - **Play**
      ```
      1) Settings -> Dashboard -> Live TV -> Tuner Devices -> Add -> M3U Tuner -> URL -> http://127.0.0.1:6464/123tv.m3u8
      2) Settings -> Dashboard -> Live TV -> TV Guide Data Providers -> Add -> XMLTV -> URL -> http://127.0.0.1:6464/tvguide.xml
      3) Settings -> Dashboard -> Scheduled Tasks -> Live TV -> Refresh Guide -> Task Triggers -> "Every 30 minutes"
      ```
    - **Note**
      ```
      Some versions does not support compressed (*.xml.gz) TV Guides.
      ```
  
  - **[IPTVnator](https://github.com/4gray/iptvnator)**
  
    <img alt="IPTVnator screenshot" width="666" src="https://user-images.githubusercontent.com/20641837/173176009-a2e86f74-46ef-464a-bbdf-9137f1d48201.jpg"/>

    Player built with [Electron](https://github.com/electron/electron) so you can run it even in browser, has light and dark themes.
    
    **Support subtitles and TV Guide.**

    - **Play**
      ```
      1) Add via URL -> http://127.0.0.1:6464/123tv.m3u8
      2) Settings -> EPG Url -> http://127.0.0.1:6464/tvguide.xml.gz
      ```
