<# : batch launcher - just double-click this file
@echo off
title One-click Webcam HLS
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ScriptFile='%~f0'; iex ([IO.File]::ReadAllText($ScriptFile))"
exit /b
#>
# ============================================================================
# One-click Webcam -> MediaMTX (HLS) -> ngrok HTTPS public link
#   Input : your ngrok authtoken (only once; press ENTER on later runs to reuse)
#   Output: public HTTPS link with the viewer login built in
# Installs everything it needs (FFmpeg, ngrok, MediaMTX) as portable copies in
#   %USERPROFILE%\Downloads\WebcamHLS  - no admin rights, no winget needed.
# ============================================================================
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # makes Invoke-WebRequest much faster
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root           = Join-Path $env:USERPROFILE 'Downloads\WebcamHLS'
$Bin            = Join-Path $Root 'bin'
$ConfigPath     = Join-Path $Root 'mediamtx.yml'
$MediaMTXOut    = Join-Path $Root 'mediamtx-stdout.log'
$MediaMTXErr    = Join-Path $Root 'mediamtx-stderr.log'
$NgrokOut       = Join-Path $Root 'ngrok-stdout.log'
$NgrokErr       = Join-Path $Root 'ngrok-stderr.log'
$FFmpegLog      = Join-Path $Root 'ffmpeg.log'
$HlsPort        = 8888
$RtspPort       = 8554
$StreamPath     = 'webcam'
$MediaMTXVer    = '1.9.3'

$Downloads = @{
    ffmpeg   = @('https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
                 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip')
    ngrok    = @('https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip')
    mediamtx = @("https://github.com/bluenviron/mediamtx/releases/download/v$MediaMTXVer/mediamtx_v$($MediaMTXVer)_windows_amd64.zip")
}

$WingetIds = @{ ffmpeg = 'Gyan.FFmpeg'; ngrok = 'Ngrok.Ngrok' }   # fallback installers

$mediamtxProc = $null; $ngrokProc = $null; $ffmpegProc = $null

function Step([string]$msg) { Write-Host "`n>> $msg" -ForegroundColor Yellow }

function Get-FileWithProgress([string]$url, [string]$out) {
    # Streams the download to disk and shows: done / total MB, %, speed, time left.
    $req = [Net.HttpWebRequest]::Create($url)
    $req.UserAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) WebcamHLS'
    $req.Timeout = 30000; $req.ReadWriteTimeout = 60000     # give up if stalled 60 s
    $resp = $req.GetResponse()
    try {
        $total = $resp.ContentLength
        $in = $resp.GetResponseStream()
        $fs = [IO.File]::Create($out)
        try {
            $buf = New-Object byte[] 262144
            $done = 0L; $lastMs = -1000L
            $sw = [Diagnostics.Stopwatch]::StartNew()
            $show = {
                $sec = [Math]::Max($sw.Elapsed.TotalSeconds, 0.001)
                $speed = $done / 1MB / $sec
                if ($total -gt 0) {
                    $pct = [int](100 * $done / $total)
                    $eta = if ($speed -gt 0) { [TimeSpan]::FromSeconds([int](($total - $done) / 1MB / $speed)).ToString('mm\:ss') } else { '--:--' }
                    $line = '   {0,6:N1} / {1:N1} MB  {2,3}%  {3:N2} MB/s  left {4}' -f ($done/1MB), ($total/1MB), $pct, $speed, $eta
                } else {
                    $line = '   {0,6:N1} MB  {1:N2} MB/s' -f ($done/1MB), $speed
                }
                Write-Host ("`r" + $line.PadRight(70)) -NoNewline
            }
            while (($n = $in.Read($buf, 0, $buf.Length)) -gt 0) {
                $fs.Write($buf, 0, $n); $done += $n
                if ($sw.ElapsedMilliseconds - $lastMs -ge 500) { $lastMs = $sw.ElapsedMilliseconds; & $show }
            }
            & $show
            Write-Host ''
            if ($total -gt 0 -and $done -ne $total) { throw "Download incomplete ($done of $total bytes)." }
        } finally { $fs.Close(); $in.Close() }
    } finally { $resp.Close() }
}

function Install-Zip([string]$name, [string[]]$urls, [string]$dest) {
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    $zip = Join-Path $env:TEMP ("webcamhls_" + [guid]::NewGuid().ToString('N') + '.zip')
    foreach ($u in $urls) {
        try {
            Write-Host "   downloading $name from $u"
            Get-FileWithProgress $u $zip
            Write-Host "   extracting..."
            Expand-Archive -Path $zip -DestinationPath $dest -Force
            return
        } catch {
            Write-Host "   failed: $($_.Exception.Message)" -ForegroundColor DarkYellow
        } finally {
            Remove-Item $zip -Force -ErrorAction SilentlyContinue
        }
    }
    throw "Could not download $name. Check your internet connection."
}

function Get-Tool([string]$name) {
    # 1) already on PATH   2) already downloaded by this script   3) download now
    $cmd = Get-Command "$name.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $local = Get-ChildItem -Path $Root -Recurse -Filter "$name.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($local) { return $local.FullName }
    try {
        Install-Zip $name $Downloads[$name] (Join-Path $Bin $name)
        $local = Get-ChildItem -Path (Join-Path $Bin $name) -Recurse -Filter "$name.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $local) { throw "$name.exe missing after extraction (antivirus may have removed it)." }
        return $local.FullName
    } catch {
        # Fallback: Windows Defender often flags the ngrok zip as "potentially unwanted"
        # and deletes it. winget installs the same tool through the package manager.
        $wid = $WingetIds[$name]
        if (-not $wid -or -not (Get-Command winget -ErrorAction SilentlyContinue)) { throw }
        Write-Host "   direct download failed ($($_.Exception.Message.Trim()))" -ForegroundColor DarkYellow
        Write-Host "   trying winget install $wid ..." -ForegroundColor Yellow
        winget install --id $wid --exact --silent --accept-package-agreements --accept-source-agreements
        $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
        $cmd = Get-Command "$name.exe" -ErrorAction SilentlyContinue
        if (-not $cmd) {
            $pk = Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') -Recurse -Filter "$name.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($pk) { return $pk.FullName }
            throw "$name could not be installed. If Windows Security blocked it, allow it in Windows Security > Protection history and run again."
        }
        return $cmd.Source
    }
}

function Wait-Port([int]$port, [int]$seconds) {
    for ($i = 0; $i -lt $seconds * 2; $i++) {
        $c = New-Object Net.Sockets.TcpClient
        try { $c.Connect('127.0.0.1', $port); return $true } catch {} finally { $c.Close() }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Show-Tail([string[]]$files) {
    foreach ($f in $files) {
        if ((Test-Path $f) -and (Get-Item $f).Length -gt 0) {
            Write-Host "---- $f ----" -ForegroundColor DarkGray
            Get-Content $f -Tail 20
        }
    }
}

function New-Password {
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    $b = New-Object byte[] 24; $rng.GetBytes($b); $rng.Dispose()
    # Alphanumeric only so it is safe inside a URL
    return ([Convert]::ToBase64String($b) -replace '[^A-Za-z0-9]', '')
}

try {
    Write-Host '=========================================' -ForegroundColor Cyan
    Write-Host '   One-click Webcam -> Public HTTPS link' -ForegroundColor Cyan
    Write-Host '=========================================' -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $Root -Force | Out-Null

    # ---- 1. Install / locate tools ---------------------------------------
    Step 'Checking required tools (first run downloads ~150 MB)...'
    $ffmpeg   = Get-Tool 'ffmpeg';   Write-Host "   FFmpeg  : $ffmpeg"
    $ngrok    = Get-Tool 'ngrok';    Write-Host "   ngrok   : $ngrok"
    $mediamtx = Get-Tool 'mediamtx'; Write-Host "   MediaMTX: $mediamtx"

    # Stop leftovers from an earlier run (ports busy / ngrok free = 1 session)
    foreach ($n in 'mediamtx','ngrok') {
        $old = Get-Process $n -ErrorAction SilentlyContinue
        if ($old) { Write-Host "   stopping old $n process..."; $old | Stop-Process -Force; Start-Sleep 1 }
    }

    # ---- 2. ngrok token (the only input) ----------------------------------
    Step 'ngrok authtoken'
    $ngrokCfg = Join-Path $env:LOCALAPPDATA 'ngrok\ngrok.yml'
    $haveSaved = (Test-Path $ngrokCfg) -and (Select-String -Path $ngrokCfg -Pattern '^\s*authtoken:\s*\S' -Quiet)
    $prompt = if ($haveSaved) { 'Paste ngrok authtoken (or press ENTER to reuse the saved one)' } else { 'Paste ngrok authtoken' }
    $sec = Read-Host $prompt -AsSecureString
    $ptr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
    try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr).Trim() }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr); $sec.Dispose() }
    if ($token) {
        & $ngrok config add-authtoken $token | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'ngrok rejected the authtoken.' }
        Write-Host '   token saved.' -ForegroundColor Green
    } elseif (-not $haveSaved) {
        throw 'An ngrok authtoken is required (https://dashboard.ngrok.com/get-started/your-authtoken).'
    } else { Write-Host '   using saved token.' -ForegroundColor Green }
    $token = $null

    # ---- 3. Webcam --------------------------------------------------------
    Step 'Detecting webcam...'
    $devOut = (& cmd.exe /c "`"$ffmpeg`" -hide_banner -list_devices true -f dshow -i dummy 2>&1" | Out-String)
    $cams = @([regex]::Matches($devOut, '"([^"]+)"\s+\(video\)') | ForEach-Object { $_.Groups[1].Value } | Select-Object -Unique)
    if ($cams.Count -eq 0) {
        Write-Host $devOut
        throw 'No webcam found. Check Settings > Privacy > Camera ("Let desktop apps access your camera").'
    }
    if ($cams.Count -eq 1) { $camera = $cams[0] }
    else {
        for ($i = 0; $i -lt $cams.Count; $i++) { Write-Host "   [$i] $($cams[$i])" }
        $sel = Read-Host 'Choose camera number (ENTER = 0)'
        if ([string]::IsNullOrWhiteSpace($sel)) { $sel = '0' }
        if ($sel -notmatch '^\d+$' -or [int]$sel -ge $cams.Count) { throw 'Invalid camera number.' }
        $camera = $cams[[int]$sel]
    }
    Write-Host "   camera: $camera" -ForegroundColor Green

    # ---- 4. MediaMTX config + start ---------------------------------------
    Step 'Starting MediaMTX (HLS server)...'
    $pubUser = 'cam_publish'; $pubPass = New-Password
    $viewUser = 'cam_view';   $viewPass = New-Password
@"
logLevel: info
rtspAddress: :$RtspPort
protocols: [tcp]
rtmp: no
webrtc: no
srt: no
hls: yes
hlsAddress: :$HlsPort
hlsVariant: lowLatency
hlsSegmentDuration: 1s
hlsSegmentCount: 7
hlsAllowOrigin: '*'
authMethod: internal
authInternalUsers:
  - user: $pubUser
    pass: "$pubPass"
    ips: []
    permissions:
      - action: publish
        path: $StreamPath
  - user: $viewUser
    pass: "$viewPass"
    ips: []
    permissions:
      - action: read
        path: $StreamPath
paths:
  ${StreamPath}:
    source: publisher
"@ | Set-Content -Path $ConfigPath -Encoding ASCII

    $mediamtxProc = Start-Process -FilePath $mediamtx -ArgumentList "`"$ConfigPath`"" -WorkingDirectory $Root `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $MediaMTXOut -RedirectStandardError $MediaMTXErr
    if (-not (Wait-Port $HlsPort 10) -or $mediamtxProc.HasExited) {
        Show-Tail @($MediaMTXOut, $MediaMTXErr)
        throw "MediaMTX failed to start (see log above). Is port $HlsPort or $RtspPort used by another app?"
    }
    Write-Host '   MediaMTX running.' -ForegroundColor Green

    # ---- 5. FFmpeg: webcam -> MediaMTX ------------------------------------
    Step 'Publishing webcam stream...'
    $rtsp = "rtsp://${pubUser}:${pubPass}@127.0.0.1:$RtspPort/$StreamPath"
    $ffArgs = "-hide_banner -nostdin -loglevel warning -f dshow -rtbufsize 100M -i `"video=$camera`" -an " +
              "-c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -b:v 1200k -maxrate 1500k -bufsize 1500k " +
              "-g 30 -keyint_min 30 -f rtsp -rtsp_transport tcp `"$rtsp`""
    $ffmpegProc = Start-Process -FilePath $ffmpeg -ArgumentList $ffArgs -PassThru -WindowStyle Hidden `
        -RedirectStandardError $FFmpegLog -RedirectStandardOutput (Join-Path $Root 'ffmpeg-stdout.log')
    Start-Sleep -Seconds 4
    if ($ffmpegProc.HasExited) {
        Show-Tail @($FFmpegLog)
        throw 'FFmpeg stopped. The camera may be in use by another app (Teams, Zoom, Camera app...).'
    }
    Write-Host '   webcam is streaming.' -ForegroundColor Green

    # ---- 6. ngrok HTTPS tunnel --------------------------------------------
    Step "Opening ngrok HTTPS tunnel to port $HlsPort..."
    $ngrokProc = Start-Process -FilePath $ngrok -ArgumentList @('http', "$HlsPort", '--log', 'stdout') `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $NgrokOut -RedirectStandardError $NgrokErr
    $publicHttps = $null
    for ($i = 0; $i -lt 30 -and -not $publicHttps; $i++) {
        Start-Sleep -Seconds 1
        if ($ngrokProc.HasExited) { break }
        try {
            $t = (Invoke-RestMethod -Uri 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 2).tunnels |
                 Where-Object { $_.public_url -like 'https://*' } | Select-Object -First 1
            if ($t) { $publicHttps = $t.public_url.TrimEnd('/') }
        } catch {}
    }
    if (-not $publicHttps) {
        Show-Tail @($NgrokOut, $NgrokErr)
        throw 'Could not get the ngrok URL. Check the token, or close ngrok running elsewhere (free plan = 1 session).'
    }

    # ---- 7. Output ---------------------------------------------------------
    $hostPart = $publicHttps -replace '^https://', ''
    $url = "https://${viewUser}:${viewPass}@$hostPart/$StreamPath/index.m3u8"
    try { Set-Clipboard -Value $url } catch {}

    Write-Host ''
    Write-Host '==================================================================' -ForegroundColor Green
    Write-Host '  WEBCAM IS LIVE  -  public HTTPS link (copied to clipboard):'      -ForegroundColor Green
    Write-Host '==================================================================' -ForegroundColor Green
    Write-Host ''
    Write-Host "  $url" -ForegroundColor Yellow
    Write-Host ''
    Write-Host '  Phone: VLC -> Open Network Stream -> paste the link.'
    Write-Host '  Login is built into the link. Anyone with it can watch - share carefully.'
    Write-Host '  The link changes every run and stops working when you close this window.'
    Write-Host ''
    Read-Host '  Press ENTER to stop streaming'
}
catch {
    Write-Host "`nERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Logs are in: $Root" -ForegroundColor DarkYellow
    Read-Host 'Press ENTER to close'
}
finally {
    foreach ($p in @($ffmpegProc, $ngrokProc, $mediamtxProc)) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    Write-Host 'Stopped.' -ForegroundColor Gray
}
