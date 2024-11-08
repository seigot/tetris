#!/bin/bash
# Colab environment setup script for Tetris

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export QT_QPA_PLATFORM=offscreen
echo "Environment setup complete. You can now run Tetris."
