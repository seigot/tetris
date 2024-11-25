#!/bin/bash
# Build script for generating documents

# Convert Markdown to PDF
pandoc templates/tech_book_fair_template.md -o output/tech_book_fair_document.pdf

echo "Document generation complete."
