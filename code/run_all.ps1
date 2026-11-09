<# =====================================================================
   HBN Brain Age Prediction Pipeline — PowerShell Launcher (Windows)
   =====================================================================
   Usage:
     .\run_all.ps1              # Run all steps (1 -> 5)
     .\run_all.ps1 1            # Run step 1 only
     .\run_all.ps1 2 3          # Run steps 2 and 3
     .\run_all.ps1 --loso       # Run all, step 3 with LOSO

   Console output is tee'd to:
     code\logs\run_YYYYMMDD_HHMMSS.log
   =====================================================================
#>

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StepArgs
)

$ErrorActionPreference = "Stop"

# -- Resolve directories ------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir    = Join-Path $ScriptDir "..\results\logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile   = Join-Path $LogDir "run_$Timestamp.log"

# -- Parse arguments ----------------------------------------------------
$RunLoso = $false
$Steps   = @()

foreach ($arg in $StepArgs) {
    switch -Regex ($arg) {
        '--loso'  { $RunLoso = $true }
        '^[1-5]$' { $Steps += [int]$arg }
        default {
            Write-Host "[ERROR] Unknown argument: $arg" -ForegroundColor Red
            Write-Host "Usage: .\run_all.ps1 [1-5 ...] [--loso]"
            exit 1
        }
    }
}

if ($Steps.Count -eq 0) {
    $Steps = 1, 2, 3, 4, 5
}

Write-Host "[INFO] Script dir:   $ScriptDir"
Write-Host "[INFO] Log file:     $LogFile"
Write-Host "[INFO] Steps to run: $($Steps -join ' ')"
Write-Host "[INFO] LOSO:          $RunLoso"
Write-Host ""

# Ensure Python output is unbuffered (so tee shows progress in real time)
$env:PYTHONUNBUFFERED = "1"

# -- Verify dependencies ------------------------------------------------
Write-Host "[INFO] Checking dependencies..."
$Packages = @("mne", "pyriemann", "sklearn", "scipy", "numpy", "pandas", "matplotlib")
$Missing  = @()
foreach ($pkg in $Packages) {
    $check = & python -c "import $pkg" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $Missing += $pkg
    }
}
if ($Missing.Count -gt 0) {
    Write-Host "[ERROR] Missing packages: $($Missing -join ' ')" -ForegroundColor Red
    Write-Host "        Run: pip install -r requirements.txt"
    exit 1
}
# pingouin is optional
$pgCheck = & python -c "import pingouin" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] pingouin not found — step04 will use fallback partial correlation."
}
Write-Host "[INFO] All critical dependencies present."
Write-Host ""

# -- Write header to log ------------------------------------------------
$header = @"
========================================================
  HBN Brain Age Pipeline — Run Log
  Timestamp: $Timestamp
  Steps:     $($Steps -join ' ')
  LOSO:      $RunLoso
  Host:      $env:COMPUTERNAME
========================================================

"@
$header | Out-File -FilePath $LogFile -Encoding UTF8

# -- Step runner --------------------------------------------------------
function Invoke-Step {
    param(
        [int]$StepNum,
        [string]$ScriptName,
        [string[]]$ExtraArgs
    )

    $scriptPath = Join-Path $ScriptDir $ScriptName
    $startTime  = Get-Date

    Write-Host ""
    Write-Host "================================================================"
    Write-Host "  Step $StepNum : $ScriptName"
    Write-Host "  Started: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    Write-Host "================================================================"

    if (-not (Test-Path $scriptPath)) {
        Write-Host "[ERROR] Script not found: $scriptPath" -ForegroundColor Red
        return $false
    }

    # Run Python script, tee output to log file
    # Use cmd /c to redirect stderr→stdout so PowerShell does not wrap
    # Python's logging (stderr) as terminating ErrorRecords.
    $argString = ($ExtraArgs -join " ")
    $cmdLine = "python `"$scriptPath`""
    if ($argString) { $cmdLine += " $argString" }
    $cmdLine += " 2>&1"
    cmd /c $cmdLine | Tee-Object -FilePath $LogFile -Append
    $exitCode = $LASTEXITCODE

    $endTime = Get-Date
    Write-Host ""
    Write-Host "[INFO] Step $StepNum finished with exit code $exitCode at $($endTime.ToString('yyyy-MM-dd HH:mm:ss'))"

    if ($exitCode -ne 0) {
        Write-Host "[ERROR] Step $StepNum FAILED (exit $exitCode)." -ForegroundColor Red
        Write-Host "[ERROR] Check log: $LogFile"
        return $false
    }
    return $true
}

# -- Execute steps ------------------------------------------------------
$OverallStart = Get-Date
$FailedStep   = ""

# Define step → script mapping
$StepMap = @{
    1 = @{ Name = "step01_preprocess.py";       Args = @("--task", "all") }
    2 = @{ Name = "step02_extract_features.py"; Args = @("--task", "all") }
    3 = @{ Name = "step03_brain_age_model.py";  Args = @("--merge-releases") }
    4 = @{ Name = "step04_statistics.py";       Args = @() }
    5 = @{ Name = "step05_figures.py";          Args = @() }
}

foreach ($step in $Steps) {
    $cfg = $StepMap[$step]

    # Special handling for step 3 with --loso
    $extraArgs = $cfg.Args
    if ($step -eq 3 -and $RunLoso) {
        $extraArgs = @("--loso")
    }

    $ok = Invoke-Step -StepNum $step -ScriptName $cfg.Name -ExtraArgs $extraArgs
    if (-not $ok) {
        $FailedStep = $step
        break
    }
}

$OverallEnd  = Get-Date
$Elapsed     = ($OverallEnd - $OverallStart)
$ElapsedMin  = [int]$Elapsed.TotalMinutes
$ElapsedSec  = $Elapsed.Seconds

# -- Summary ------------------------------------------------------------
Write-Host ""
Write-Host "========================================================"
Write-Host "  Pipeline Summary"
Write-Host "  Total elapsed: ${ElapsedMin}m ${ElapsedSec}s"
Write-Host "  Log file:      $LogFile"
if ($FailedStep -ne "") {
    Write-Host "  Status:        FAILED at step $FailedStep" -ForegroundColor Red
    Write-Host "========================================================"
    exit 1
} else {
    Write-Host "  Status:        ALL STEPS COMPLETED" -ForegroundColor Green
    Write-Host "========================================================"
}

exit 0
