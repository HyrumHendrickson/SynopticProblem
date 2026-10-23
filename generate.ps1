<#
.SYNOPSIS
    Regenerates the examples folder for the Synoptic Problem explorer.

.DESCRIPTION
    Runs overlap/code.py and stats/code.py for every combination of:

        window          5, 10, 15, 20
        threshold       ceil(window / 3) up to window (inclusive)
        word order      off (ignored) and on (respected, longest common subsequence)
        compared on     lemma and norm
        content only    on (function words excluded) and off (all words)

    Output layout:

        examples/Window10/Threshold05/Lemma_ContentOnly_OrderOff/
            synoptic_overlaps.txt
            synoptic_stats.txt
            media/01_*.png ... 22_*.png

    A manifest (examples/manifest.js) is written at the end. index.html reads it
    to know which combinations exist and to show section counts and directional
    overlap without opening each report.

    Each run is capped by a time limit and a memory limit. Combinations that
    exceed either limit are killed, removed, and listed in
    examples/generation_report.txt. Very low thresholds, especially with content
    only turned off, can produce tens of millions of matching windows and are
    the usual cause. Completed combinations are skipped on re-run, so the
    script can be interrupted and resumed. Use -Force to regenerate everything.

.PARAMETER Windows
    Window sizes to generate. Default: 5, 10, 15, 20.

.PARAMETER MaxParallel
    Number of Python processes to run at once.

.PARAMETER TimeoutMinutes
    Time limit for each step (overlap, then stats) of each combination.

.PARAMETER MaxMemoryGB
    Memory limit for each Python process.

.PARAMETER Python
    Path to a Python 3 executable. Detected automatically if omitted.

.PARAMETER Force
    Regenerate combinations that already exist.

.PARAMETER SkipOverlapText
    Delete synoptic_overlaps.txt after the statistics are produced, to save
    disk space. Some low-threshold overlap files exceed 10 MB each.

.PARAMETER ListOnly
    Print the number of combinations and exit.

.PARAMETER First
    Process only the first N queued combinations (testing aid). Combinations
    are queued from highest to lowest threshold-to-window ratio, so the
    fastest ones run first.

.EXAMPLE
    .\generate_examples.ps1

.EXAMPLE
    .\generate_examples.ps1 -MaxParallel 8 -TimeoutMinutes 30
#>
[CmdletBinding()]
param(
    [int[]]$Windows = @(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    [int]$MaxParallel = [Math]::Max(1, [Math]::Min(8, [Environment]::ProcessorCount - 1)),
    [double]$TimeoutMinutes = 15,
    [double]$MaxMemoryGB = 6,
    [string]$Python = '',
    [switch]$Force,
    [switch]$SkipOverlapText,
    [switch]$ListOnly,
    [int]$First = 0
)

$ErrorActionPreference = 'Stop'
$Inv = [Globalization.CultureInfo]::InvariantCulture
$ExpectedImages = 21

$Root = $PSScriptRoot
if (-not $Root) { $Root = (Get-Location).Path }
$OverlapScript = Join-Path $Root 'overlap/code.py'
$StatsScript   = Join-Path $Root 'stats/code.py'
$ExamplesDir   = Join-Path $Root 'examples'
$TextDir       = Join-Path $Root 'Text/sblgnt'

foreach ($p in @($OverlapScript, $StatsScript, $TextDir)) {
    if (-not (Test-Path $p)) {
        throw "Not found: $p. Place this script at the repository root."
    }
}

# --------------------------------------------------------------- combinations
function New-Combo($w, $t, $field, $content, $order) {
    $fieldName   = (Get-Culture).TextInfo.ToTitleCase($field)
    $contentName = if ($content) { 'ContentOnly' } else { 'AllWords' }
    $orderName   = if ($order)   { 'OrderOn' }     else { 'OrderOff' }
    $variant = '{0}_{1}_{2}' -f $fieldName, $contentName, $orderName
    $dir = Join-Path $ExamplesDir ('Window{0:00}/Threshold{1:00}/{2}' -f $w, $t, $variant)
    [pscustomobject]@{
        W           = $w
        T           = $t
        Field       = $field
        Content     = $content
        Order       = $order
        Key         = '{0}-{1}-{2}-{3}-{4}' -f $w, $t, $field, [int]$content, [int]$order
        Label       = 'W{0,-2} T{1,-2} {2}' -f $w, $t, $variant
        Dir         = $dir
        RelDir      = ('examples/Window{0:00}/Threshold{1:00}/{2}' -f $w, $t, $variant)
        OverlapFile = Join-Path $dir 'synoptic_overlaps.txt'
        StatsFile   = Join-Path $dir 'synoptic_stats.txt'
        MediaDir    = Join-Path $dir 'media'
    }
}

$all = New-Object System.Collections.ArrayList
foreach ($w in $Windows) {
    $tMin = [int][Math]::Ceiling($w / 3.0)
    foreach ($t in $tMin..$w) {
        foreach ($field in 'lemma', 'norm') {
            foreach ($content in $true, $false) {
                foreach ($order in $false, $true) {
                    [void]$all.Add((New-Combo $w $t $field $content $order))
                }
            }
        }
    }
}
Write-Host ("{0} combinations in total." -f $all.Count)
if ($ListOnly) { return }

# ------------------------------------------------------------------- python
function Find-Python {
    $candidates = New-Object System.Collections.ArrayList
    if ($Python) { [void]$candidates.Add(@($Python)) }
    [void]$candidates.Add(@('python'))
    [void]$candidates.Add(@('python3'))
    [void]$candidates.Add(@('py', '-3'))
    foreach ($c in $candidates) {
        $exe = $c[0]
        $pre = @()
        if ($c.Count -gt 1) { $pre = $c[1..($c.Count - 1)] }
        $cmd = Get-Command $exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        try {
            $out = & $cmd.Source @pre -c "import sys, numpy, matplotlib; print(sys.version_info[0])" 2>$null
            if ($LASTEXITCODE -eq 0 -and "$out".Trim() -eq '3') {
                return @{ Exe = $cmd.Source; Pre = $pre }
            }
        } catch { }
    }
    throw "No Python 3 with numpy and matplotlib found. Install with: pip install numpy matplotlib (or pass -Python <path>)."
}
$Py = Find-Python
Write-Host ("Python: {0} {1}" -f $Py.Exe, ($Py.Pre -join ' '))

$env:PYTHONIOENCODING = 'utf-8'
$env:MPLBACKEND = 'Agg'      # headless; avoids GUI backends in parallel runs

$LogDir = Join-Path ([IO.Path]::GetTempPath()) 'synoptic_example_logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ----------------------------------------------------------------- helpers
function Test-Complete($c) {
    if (-not (Test-Path $c.StatsFile)) { return $false }
    if (-not (Test-Path $c.MediaDir)) { return $false }
    return ((Get-ChildItem -Path $c.MediaDir -Filter '*.png').Count -ge $ExpectedImages)
}

function Test-HasSections($file) {
    if (-not (Test-Path $file)) { return $false }
    return [bool](Select-String -Path $file -Pattern '^=== ' -Quiet)
}

function Test-EmptyResult($c) {
    # The overlap step ran and found no sections, so there is nothing to chart.
    return ((Test-Path $c.OverlapFile) -and -not (Test-Path $c.StatsFile) -and -not (Test-HasSections $c.OverlapFile))
}

function Get-LastLine($path) {
    if (-not (Test-Path $path)) { return '' }
    $l = Get-Content -Path $path -Tail 3 -ErrorAction SilentlyContinue | Where-Object { $_.Trim() }
    if ($l) { return ($l | Select-Object -Last 1).Trim() }
    return ''
}

function Start-Step($task, $step) {
    $c = $task.Combo
    $log = Join-Path $LogDir ('{0}.{1}' -f $c.Key, $step)
    $q = { param($s) '"' + $s + '"' }
    if ($step -eq 'overlap') {
        $argList = @($Py.Pre) + @(
            (& $q $OverlapScript),
            '--window', $c.W, '--threshold', $c.T, '--field', $c.Field,
            $(if ($c.Content) { '--content-only' } else { '--no-content-only' }),
            $(if ($c.Order) { '--no-ignore-word-order' } else { '--ignore-word-order' }),
            '--out', (& $q $c.OverlapFile)
        )
    } else {
        $argList = @($Py.Pre) + @(
            (& $q $StatsScript),
            '--overlap', (& $q $c.OverlapFile),
            '--out', (& $q $c.StatsFile),
            '--media', (& $q $c.MediaDir)
        )
    }
    $proc = Start-Process -FilePath $Py.Exe -ArgumentList ($argList -join ' ') `
        -NoNewWindow -PassThru `
        -RedirectStandardOutput ($log + '.out') -RedirectStandardError ($log + '.err')
    $null = $proc.Handle   # keeps ExitCode readable after exit
    $task.Proc = $proc
    $task.Step = $step
    $task.StepStart = Get-Date
    $task.ErrLog = $log + '.err'
}

$Failures = New-Object System.Collections.ArrayList
$Counts = @{ Done = 0; Skipped = 0; Empty = 0; Failed = 0 }
$Total = 0
$Finished = 0

function Complete-Task($task, $status, $detail) {
    $script:Finished++
    $c = $task.Combo
    $secs = [int]((Get-Date) - $task.Start).TotalSeconds
    switch ($status) {
        'OK'    { $Counts.Done++ }
        'EMPTY' { $Counts.Empty++ }
        'FAIL'  {
            $Counts.Failed++
            if (Test-Path $c.Dir) { Remove-Item -Recurse -Force $c.Dir -ErrorAction SilentlyContinue }
            [void]$Failures.Add(('{0} : {1}' -f $c.RelDir, $detail))
        }
    }
    $msg = '[{0,3}/{1}] {2,-5} {3} ({4}s){5}' -f $script:Finished, $script:Total, $status, $c.Label, $secs, $(if ($detail) { ' - ' + $detail } else { '' })
    $color = switch ($status) { 'OK' { 'Green' } 'EMPTY' { 'Yellow' } default { 'Red' } }
    Write-Host $msg -ForegroundColor $color
}

# ------------------------------------------------------------------- queue
$queue = New-Object System.Collections.Generic.Queue[object]
$toRun = $all | Sort-Object @{ Expression = { - ($_.T / [double]$_.W) } }, W, T
foreach ($c in $toRun) {
    if (-not $Force -and ((Test-Complete $c) -or (Test-EmptyResult $c))) {
        $Counts.Skipped++
        continue
    }
    $queue.Enqueue($c)
}
if ($First -gt 0) {
    $limited = New-Object System.Collections.Generic.Queue[object]
    for ($i = 0; $i -lt $First -and $queue.Count -gt 0; $i++) { $limited.Enqueue($queue.Dequeue()) }
    $queue = $limited
}
$Total = $queue.Count
Write-Host ("{0} to run, {1} already complete. Parallel: {2}. Limits per step: {3} min, {4} GB." -f `
        $Total, $Counts.Skipped, $MaxParallel, $TimeoutMinutes, $MaxMemoryGB)

# -------------------------------------------------------------------- run
$active = New-Object System.Collections.ArrayList
$clock = Get-Date
try {
    while ($queue.Count -gt 0 -or $active.Count -gt 0) {
        while ($active.Count -lt $MaxParallel -and $queue.Count -gt 0) {
            $c = $queue.Dequeue()
            if (Test-Path $c.Dir) { Remove-Item -Recurse -Force $c.Dir }
            New-Item -ItemType Directory -Force -Path $c.Dir | Out-Null
            $task = [pscustomobject]@{
                Combo = $c; Proc = $null; Step = ''; StepStart = $null; ErrLog = ''; Start = Get-Date
            }
            Start-Step $task 'overlap'
            [void]$active.Add($task)
        }

        Start-Sleep -Milliseconds 500

        foreach ($task in @($active)) {
            $p = $task.Proc
            $c = $task.Combo
            if (-not $p.HasExited) {
                $elapsed = ((Get-Date) - $task.StepStart).TotalMinutes
                $mem = 0
                try { $p.Refresh(); $mem = $p.WorkingSet64 } catch { }
                $reason = ''
                if ($elapsed -gt $TimeoutMinutes) {
                    $reason = 'exceeded {0} min during {1}' -f $TimeoutMinutes, $task.Step
                } elseif ($mem -gt ($MaxMemoryGB * 1GB)) {
                    $reason = 'exceeded {0} GB during {1}' -f $MaxMemoryGB, $task.Step
                }
                if ($reason) {
                    try { $p.Kill() } catch { }
                    [void]$active.Remove($task)
                    Complete-Task $task 'FAIL' $reason
                }
                continue
            }

            if ($p.ExitCode -ne 0) {
                [void]$active.Remove($task)
                $why = Get-LastLine $task.ErrLog
                if (-not $why) { $why = 'exit code ' + $p.ExitCode }
                Complete-Task $task 'FAIL' ('{0} failed: {1}' -f $task.Step, $why)
                continue
            }

            if ($task.Step -eq 'overlap') {
                if (-not (Test-HasSections $c.OverlapFile)) {
                    [void]$active.Remove($task)
                    Complete-Task $task 'EMPTY' 'no sections at these settings'
                    continue
                }
                Start-Step $task 'stats'
                continue
            }

            # stats step finished
            [void]$active.Remove($task)
            if (Test-Complete $c) {
                if ($SkipOverlapText) { Remove-Item -Force $c.OverlapFile -ErrorAction SilentlyContinue }
                Complete-Task $task 'OK' ''
            } else {
                Complete-Task $task 'FAIL' 'stats finished but images are missing'
            }
        }
    }
}
finally {
    foreach ($task in @($active)) {
        try { if (-not $task.Proc.HasExited) { $task.Proc.Kill() } } catch { }
    }
}

# ------------------------------------------------- prune empty directories
if (Test-Path $ExamplesDir) {
    Get-ChildItem -Path $ExamplesDir -Recurse -Directory |
        Sort-Object { $_.FullName.Length } -Descending |
        ForEach-Object {
            if (-not (Get-ChildItem -Path $_.FullName -Force)) { Remove-Item -Force $_.FullName }
        }
}

# ---------------------------------------------------------------- manifest
function Read-StatsSummary($c) {
    $lines = Get-Content -Path $c.StatsFile -TotalCount 60
    $sections = $null
    $heads = [ordered]@{}
    foreach ($line in $lines) {
        if ($line -match 'Sections read:\s*(\d+)\s*\(Mt-Mk (\d+), Mt-Lk (\d+), Mk-Lk (\d+)\)') {
            $sections = @($Matches[1], $Matches[2], $Matches[3], $Matches[4])
        } elseif ($line -match '^\s*(Matthew|Mark|Luke)\s+in\s+(Matthew|Mark|Luke)\s*:\s*([0-9.]+)%') {
            $heads[('{0}>{1}' -f $Matches[1], $Matches[2])] = $Matches[3]
        }
    }
    if (-not $sections) { return $null }
    return @{ Sections = $sections; Heads = $heads }
}

$entries = New-Object System.Collections.ArrayList
$available = 0
foreach ($c in $all) {
    $json = $null
    if (Test-Complete $c) {
        $s = Read-StatsSummary $c
        if ($s) {
            $h = ($s.Heads.GetEnumerator() | ForEach-Object { '"{0}":{1}' -f $_.Key, $_.Value }) -join ','
            $json = '"{0}":{{"dir":"{1}","sections":[{2}],"heads":{{{3}}}}}' -f $c.Key, $c.RelDir, ($s.Sections -join ','), $h
        }
    } elseif (Test-EmptyResult $c) {
        $json = '"{0}":{{"dir":"{1}","empty":true}}' -f $c.Key, $c.RelDir
    }
    if ($json) { [void]$entries.Add($json); $available++ }
}
$manifest = 'window.SYNOPTIC_MANIFEST={{"generated":"{0}","expected":{1},"combos":{{{2}}}}};' -f `
    (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'), $all.Count, ($entries -join ',')
New-Item -ItemType Directory -Force -Path $ExamplesDir | Out-Null
[IO.File]::WriteAllText((Join-Path $ExamplesDir 'manifest.js'), $manifest, (New-Object Text.UTF8Encoding($false)))

# ------------------------------------------------------------------ report
$report = New-Object System.Collections.ArrayList
[void]$report.Add('Generated: ' + (Get-Date).ToString('s'))
[void]$report.Add(('Available in manifest: {0} of {1}' -f $available, $all.Count))
[void]$report.Add(('This run: {0} generated, {1} empty, {2} failed, {3} skipped (already complete)' -f $Counts.Done, $Counts.Empty, $Counts.Failed, $Counts.Skipped))
if ($Failures.Count -gt 0) {
    [void]$report.Add('')
    [void]$report.Add(('Failed in this run (limits: {0} min, {1} GB):' -f $TimeoutMinutes, $MaxMemoryGB))
    foreach ($f in $Failures) { [void]$report.Add('  ' + $f) }
}
Set-Content -Path (Join-Path $ExamplesDir 'generation_report.txt') -Value $report -Encoding UTF8

$sizeMB = [Math]::Round(((Get-ChildItem -Path $ExamplesDir -Recurse -File | Measure-Object Length -Sum).Sum) / 1MB, 0)
Write-Host ''
Write-Host ('Finished in {0:N1} min. {1} of {2} combinations available, {3} failed, {4} empty. examples/ is {5} MB.' -f `
        ((Get-Date) - $clock).TotalMinutes, $available, $all.Count, $Counts.Failed, $Counts.Empty, $sizeMB)
if ($Failures.Count -gt 0) {
    Write-Host 'Failures are listed in examples/generation_report.txt. Re-run to retry them, optionally with a larger -TimeoutMinutes.' -ForegroundColor Yellow
}
