param(
    [string]$DownloadPath = (Join-Path $env:USERPROFILE "Downloads\ebm-tail-audit-iclr2026-anonymous.pdf"),
    [switch]$KeepAux
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$PaperDir = Join-Path $RepoRoot "paper\iclr2026"
$BuildDir = Join-Path $PaperDir "build"

New-Item -ItemType Directory -Force $BuildDir | Out-Null
New-Item -ItemType Directory -Force (Split-Path -Parent $DownloadPath) | Out-Null

Push-Location $PaperDir
try {
    $builtWithLatexmk = $false
    $latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
    $perl = Get-Command perl -ErrorAction SilentlyContinue
    if ($latexmk -and $perl) {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -eq 0) {
            $builtWithLatexmk = $true
        } else {
            Write-Warning "latexmk failed; falling back to pdflatex/bibtex."
        }
    } elseif ($latexmk) {
        Write-Warning "latexmk is installed but Perl is unavailable; falling back to pdflatex/bibtex."
    }

    if (-not $builtWithLatexmk) {
        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed." }

        & bibtex build/main
        if ($LASTEXITCODE -ne 0) { throw "bibtex failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 2 failed." }

        & pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
        if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 3 failed." }
    }
} finally {
    Pop-Location
}

$PdfPath = Join-Path $BuildDir "main.pdf"
if (-not (Test-Path $PdfPath)) {
    throw "Expected PDF was not produced: $PdfPath"
}

Copy-Item -LiteralPath $PdfPath -Destination $DownloadPath -Force

if (-not $KeepAux) {
    Get-ChildItem -LiteralPath $BuildDir -File |
        Where-Object { $_.Name -ne "main.pdf" } |
        Remove-Item -Force
}

Write-Host "Built $PdfPath"
Write-Host "Copied anonymous PDF to $DownloadPath"
