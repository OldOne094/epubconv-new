if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Host "Python launcher 'py' not found. Install Python 3.11 or 3.12 first."
    exit 1
}

py -3.12 -m venv .venv
if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Could not create a Python 3.12 virtual environment (py -3.12 --version to check it's installed)."
    exit 1
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Base install failed (pip install -e .[dev]) - see the errors above."
    exit 1
}

Write-Host ""
Write-Host "Checking for an NVIDIA GPU to enable acceleration (optional; falls back to CPU on any doubt)..."

$pip = ".\.venv\Scripts\pip.exe"
$python = ".\.venv\Scripts\python.exe"
$gpuVerified = $false

# Deliberately not using $ErrorActionPreference = "Stop" here: in Windows
# PowerShell 5.1, any native command (pip, python, nvidia-smi, ...) that
# writes so much as one harmless informational line to stderr gets promoted
# into a terminating error under that setting - even on a fully successful
# run - which aborted this whole script before this was found. Every native
# call below is checked explicitly via $LASTEXITCODE / its actual output
# instead of relying on exceptions for control flow.
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $smiOutput = & nvidia-smi 2>$null
        $cudaMatch = ($smiOutput | Select-String -Pattern "CUDA Version:\s*([\d.]+)").Matches
        if ($cudaMatch.Count -gt 0) {
            $cudaVersion = [version]$cudaMatch[0].Groups[1].Value
            # Three channels we've actually verified paddlepaddle-gpu==3.2.1 ships for
            # (see README) - pick the highest one the installed driver supports.
            $channel = $null
            if ($cudaVersion -ge [version]"12.9") { $channel = "cu129" }
            elseif ($cudaVersion -ge [version]"12.6") { $channel = "cu126" }
            elseif ($cudaVersion -ge [version]"11.8") { $channel = "cu118" }

            if ($channel) {
                Write-Host "NVIDIA GPU found (driver supports up to CUDA $cudaVersion) - trying the $channel build..."
                # paddlepaddle (CPU) and paddlepaddle-gpu both provide the same `paddle`
                # import - having both installed at once is a known source of silent
                # breakage, so the CPU one has to go before the GPU one goes in.
                & $pip uninstall -y paddlepaddle *> $null
                & $pip install "paddlepaddle-gpu==3.2.1" -i "https://www.paddlepaddle.org.cn/packages/stable/$channel/" --quiet 2>$null
                if ($LASTEXITCODE -eq 0) {
                    # Never trust "installed without error" alone - a build that's not
                    # actually compatible with this specific GPU can still import fine
                    # and silently return wrong numbers. Verify real computation.
                    $checkOutput = & $python -c "import paddle; x = paddle.to_tensor([1.0, 2.0, 3.0]); paddle.set_device('gpu'); print((x * 2).numpy())" 2>$null
                    if ($checkOutput -match "2\.\s*4\.\s*6\.") {
                        Write-Host "GPU acceleration verified and working ($channel)."
                        $gpuVerified = $true
                    } else {
                        Write-Host "GPU build installed but the compute check came back wrong - falling back to CPU."
                    }
                } else {
                    Write-Host "Installing the GPU build failed - falling back to CPU."
                }
            } else {
                Write-Host "NVIDIA driver found, but it only supports CUDA $cudaVersion (too old for a supported build here) - using CPU."
            }
        } else {
            Write-Host "nvidia-smi ran but its CUDA version couldn't be read - using CPU."
        }
    } else {
        Write-Host "No NVIDIA GPU detected - using CPU. That's completely normal; the tool works the same either way, just slower."
    }
} catch {
    Write-Host "GPU detection hit an unexpected error ($_) - using CPU."
}

if (-not $gpuVerified) {
    # Single cleanup point instead of one per failure branch above, so this
    # still runs correctly even if something throws mid-way through the GPU
    # steps (caught by the try/catch above) rather than failing a specific
    # check - paddlepaddle-gpu can end up "installed" (pip exit 0) without
    # ever being verified working, and it must not coexist with plain
    # paddlepaddle no matter which path got us here. Both lines are no-ops
    # if the package in question was never installed in the first place.
    & $pip uninstall -y paddlepaddle-gpu *> $null
    & $pip install paddlepaddle --quiet *> $null
}

# A double-clickable launcher, so opening the program again later never needs
# a terminal: it just starts epubconv-review with no book pre-selected (pick
# one in the browser - upload, or type any path) and opens the browser itself.
$launcherPath = Join-Path $PSScriptRoot "تشغيل البرنامج.cmd"
@"
@echo off
cd /d "%~dp0"
start "" ".venv\Scripts\epubconv-review.exe"
"@ | Set-Content -Path $launcherPath -Encoding ASCII

Write-Host ""
Write-Host "Done. Launch anytime by double-clicking: تشغيل البرنامج.cmd"
Write-Host "(or manually: .\.venv\Scripts\Activate.ps1, then epubconv-review)"
Write-Host ""
Write-Host "Starting it now..."
Start-Process -FilePath $launcherPath
