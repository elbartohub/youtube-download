#!/bin/bash
"""
Run script for YouTube Downloader with Translation
"""

echo "🚀 Starting YouTube Downloader with Translation..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    echo "   Run: ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if required packages are installed
python -c "import gradio, transformers, torch, yt_dlp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Required packages not installed. Please run setup.sh first."
    echo "   Run: ./setup.sh"
    exit 1
fi

# Run the application
echo "🌐 Launching web interface..."
echo "📱 Opening browser at: http://localhost:7860"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

python youtube_gradio_app.py
