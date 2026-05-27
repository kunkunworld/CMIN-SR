# Local LaTeX Writing Workspace

This folder is the local writing workspace for the CMIN-SR paper.

## Do I Need a Python/Conda Env?

Not for normal LaTeX writing.

LaTeX is usually a system-level tool, not a Python environment. You mainly need
one of:

- MiKTeX on Windows
- TeX Live
- Tectonic

Python is only useful if you want to regenerate figures/tables from CSV. The
paper source itself can be edited and compiled without Python.

## Recommended Windows Setup

Install one LaTeX distribution:

1. MiKTeX: https://miktex.org/download
2. Or TeX Live: https://tug.org/texlive/

Recommended editor:

- VS Code
- VS Code extension: LaTeX Workshop

After installation, restart PowerShell / VS Code and check:

```powershell
pdflatex --version
latexmk --version
```

If `latexmk` is missing but `pdflatex` works, use:

```powershell
.\build_pdf.ps1 -Engine pdflatex
```

## Main Files

- `main.tex`: main paper file.
- `sections/`: section drafts.
- `figures/`: paper figures.
- `latex_tables/`: LaTeX table snippets.
- `tables/`: CSV source tables.
- `notes/`: writing notes and project summary.
- `ARTICLE_STRUCTURE_CN.md`: Chinese writing guide.

## Build

Preferred:

```powershell
.\build_pdf.ps1
```

Fallback:

```powershell
.\build_pdf.ps1 -Engine pdflatex
```

Clean auxiliary files:

```powershell
.\clean_latex.ps1
```

## Collaboration

If you want version control later:

```powershell
git init
git add .
git commit -m "Initial CMIN-SR paper draft"
```

Do not commit large generated logs unless needed. The `.gitignore` already
ignores common LaTeX build artifacts.

