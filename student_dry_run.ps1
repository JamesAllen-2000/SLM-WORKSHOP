<#
=============================================================================
 STUDENT DRY RUN - walk the participant path by hand        [ ORGANISER ]
=============================================================================
 Runs exactly what a participant runs, in order, pausing between demos so you
 can actually LOOK at each answer.

     .\student_dry_run.ps1              # full run, interactive demos
     .\student_dry_run.ps1 -Reset       # first move existing build output aside
     .\student_dry_run.ps1 -DemosOnly   # skip setup checks, just the 5 demos

 Use smoke_test.py instead if you want an unattended pass/fail.
 Use THIS when you want to see what the room will see.
=============================================================================
#>
param(
    [switch]$Reset,
    [switch]$DemosOnly
)

$ErrorActionPreference = "Continue"

function Step {
    param([string]$Title, [string]$Command, [string]$Look)

    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    if ($Look) {
        Write-Host "  LOOK FOR: $Look" -ForegroundColor Yellow
    }
    Write-Host "  `$ $Command" -ForegroundColor DarkGray
    Write-Host ""

    $start = Get-Date
    Invoke-Expression $Command
    $secs = ((Get-Date) - $start).TotalSeconds

    Write-Host ""
    if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) {
        Write-Host ("  [OK] finished in {0:N0}s" -f $secs) -ForegroundColor Green
    } else {
        Write-Host ("  [FAILED] exit code $LASTEXITCODE after {0:N0}s" -f $secs) -ForegroundColor Red
    }
    Write-Host "  Press Enter for the next step..." -ForegroundColor DarkGray
    Read-Host | Out-Null
}

# ---------------------------------------------------------------------------
# OPTIONAL RESET - put the repo back to a participant's starting state
# ---------------------------------------------------------------------------
if ($Reset) {
    Write-Host "Moving build output aside (reversible - nothing is deleted)..." -ForegroundColor Yellow
    foreach ($d in @("chroma_db", "results", "analysis")) {
        if (Test-Path $d) {
            $bak = "_bak_$d"
            if (Test-Path $bak) { Remove-Item -Recurse -Force $bak }
            Move-Item $d $bak
            Write-Host "  $d -> _bak_$d"
        }
    }
    Write-Host "Done. Restore later with: Move-Item _bak_<name> <name>" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host ""
Write-Host "STUDENT DRY RUN" -ForegroundColor White
Write-Host "Walking the participant path exactly as written in WORKSHOP_GUIDE.md"
Write-Host ""

# ---------------------------------------------------------------------------
# PART A - what a participant does at home
# ---------------------------------------------------------------------------
if (-not $DemosOnly) {
    Step -Title "PART A - Verify setup" `
         -Command "python verify_setup.py" `
         -Look "Every line OK. The vector database may say MISSING - that is expected before B1."

    Step -Title "PART B0 - Hardware warm-up" `
         -Command "python inspect_system.py" `
         -Look "Accelerators: none - CPU only. That is the premise of the workshop."

    Step -Title "PART B1 - Build the RAG knowledge base" `
         -Command "python 01_rag_baseline.py" `
         -Look "'RAG knowledge base ready: 592 chunks' in about a minute."
}

# ---------------------------------------------------------------------------
# PART B - the five demos
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("*" * 70) -ForegroundColor Magenta
Write-Host "  THE FIVE DEMOS" -ForegroundColor Magenta
Write-Host "  In each one, type this question before you type 'exit':" -ForegroundColor Magenta
Write-Host "     What did the last technician find when they serviced P-104?" -ForegroundColor White
Write-Host ("*" * 70) -ForegroundColor Magenta

Step -Title "DEMO 1 - Base model (no fine-tuning, no RAG)" `
     -Command "python tests\test_00_base_slm.py" `
     -Look "~6 t/s. Watch it list causes then STOP without an action plan. Ask the P-104 question - it will invent an answer."

Step -Title "DEMO 2 - Fine-tuned" `
     -Command "python tests\test_02_adapter.py" `
     -Look "Same ~6 t/s, but it now reaches [ ACTION PLAN ] and is noticeably terser than demo 1."

Step -Title "DEMO 3 - Fine-tuned + RAG" `
     -Command "python tests\test_03_adapter_rag.py" `
     -Look "Real values and a 'Sources used:' list. Ask the P-104 question again - now it can answer."

Step -Title "DEMO 4 - Quantized (4-bit GGUF)" `
     -Command "python tests\test_04_gguf.py" `
     -Look "~30 t/s - about 5x faster than demos 1-3. Backend must say CPU."

Step -Title "DEMO 5 - Quantized + RAG (the finished product)" `
     -Command "python 04_gguf_rag.py" `
     -Look "Fast AND grounded. Check 'RAG mode' and the source list. Try turning wifi off."

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host "  DRY RUN COMPLETE" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Green
Write-Host ""
Write-Host "Check the logs each demo wrote:"
Write-Host "  results\test_00_base_log.txt"
Write-Host "  results\test_02_adapter_log.txt"
Write-Host "  results\test_03_adapter_rag_log.txt"
Write-Host "  results\test_04_gguf_log.txt"
Write-Host "  results\04_gguf_rag_log.txt"
Write-Host ""
if ($Reset) {
    Write-Host "Restore your previous build output with:" -ForegroundColor Yellow
    Write-Host "  Move-Item _bak_chroma_db chroma_db   (etc.)" -ForegroundColor Yellow
}
