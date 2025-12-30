#!/bin/bash

# Compilation script for research presentation PDF
# This script compiles the LaTeX document to PDF

echo "========================================="
echo "Clinical Valence Testing - PDF Compilation"
echo "========================================="
echo ""

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo ""
    echo "Please install LaTeX:"
    echo "  Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "  macOS: brew install --cask mactex"
    echo "  Windows: Install MiKTeX or TeX Live"
    echo ""
    exit 1
fi

echo "[OK] pdflatex found"
echo ""

# Compile the document
echo "Compiling RESEARCH_PRESENTATION.tex..."
echo ""

# First pass
echo "[1/3] First pass (generating references)..."
pdflatex -interaction=nonstopmode RESEARCH_PRESENTATION.tex > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: First compilation pass failed!"
    pdflatex -interaction=nonstopmode RESEARCH_PRESENTATION.tex
    exit 1
fi

# Second pass (for references)
echo "[2/3] Second pass (resolving references)..."
pdflatex -interaction=nonstopmode RESEARCH_PRESENTATION.tex > /dev/null 2>&1

# Third pass (final)
echo "[3/3] Third pass (finalizing)..."
pdflatex -interaction=nonstopmode RESEARCH_PRESENTATION.tex > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "[SUCCESS] Compilation complete"
    echo "========================================="
    echo ""
    echo "PDF created: RESEARCH_PRESENTATION.pdf"
    echo ""
    echo "Output files:"
    ls -lh RESEARCH_PRESENTATION.pdf 2>/dev/null
    echo ""

    # Clean up auxiliary files
    echo "Cleaning up auxiliary files..."
    rm -f RESEARCH_PRESENTATION.aux \
          RESEARCH_PRESENTATION.log \
          RESEARCH_PRESENTATION.out \
          RESEARCH_PRESENTATION.toc \
          RESEARCH_PRESENTATION.bbl \
          RESEARCH_PRESENTATION.blg

    echo "[OK] Cleanup complete"
    echo ""
    echo "You can now view the PDF:"
    echo "  Linux: xdg-open RESEARCH_PRESENTATION.pdf"
    echo "  macOS: open RESEARCH_PRESENTATION.pdf"
    echo "  Windows: start RESEARCH_PRESENTATION.pdf"
    echo ""
else
    echo ""
    echo "ERROR: Compilation failed!"
    echo "Check the log file for details:"
    echo "  cat RESEARCH_PRESENTATION.log"
    echo ""
    exit 1
fi
