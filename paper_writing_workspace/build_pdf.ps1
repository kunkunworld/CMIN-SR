param(
    [string]$Main = "main.tex",
    [ValidateSet("latexmk", "pdflatex", "xelatex")]
    [string]$Engine = "latexmk"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Require-Command($name) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "Missing command: $name" -ForegroundColor Red
        Write-Host "Install MiKTeX or TeX Live, then restart PowerShell/VS Code." -ForegroundColor Yellow
        exit 1
    }
}

function Run-Bibtex-IfNeeded($mainFile) {
    $base = [System.IO.Path]::GetFileNameWithoutExtension($mainFile)
    $aux = "$base.aux"
    if (Test-Path $aux) {
        $auxText = Get-Content $aux -Raw
        if ($auxText -match "\\citation") {
            $bib = Get-Command bibtex -ErrorAction SilentlyContinue
            if ($bib) {
                bibtex $base
            } else {
                Write-Host "bibtex not found; skipping bibliography pass." -ForegroundColor Yellow
            }
        } else {
            Write-Host "No citations found; skipping bibtex." -ForegroundColor Yellow
        }
    }
}

if ($Engine -eq "latexmk") {
    Require-Command "latexmk"
    latexmk -pdf -interaction=nonstopmode -halt-on-error $Main
} elseif ($Engine -eq "pdflatex") {
    Require-Command "pdflatex"
    pdflatex -interaction=nonstopmode -halt-on-error $Main
    Run-Bibtex-IfNeeded $Main
    pdflatex -interaction=nonstopmode -halt-on-error $Main
    pdflatex -interaction=nonstopmode -halt-on-error $Main
} elseif ($Engine -eq "xelatex") {
    Require-Command "xelatex"
    xelatex -interaction=nonstopmode -halt-on-error $Main
    Run-Bibtex-IfNeeded $Main
    xelatex -interaction=nonstopmode -halt-on-error $Main
    xelatex -interaction=nonstopmode -halt-on-error $Main
}

Write-Host "Build finished." -ForegroundColor Green
