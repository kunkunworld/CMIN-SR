$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$patterns = @(
    "*.aux", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.log",
    "*.out", "*.synctex.gz", "*.toc", "*.lof", "*.lot"
)

foreach ($pattern in $patterns) {
    Get-ChildItem -Path . -Recurse -Filter $pattern -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

Write-Host "LaTeX auxiliary files cleaned." -ForegroundColor Green

