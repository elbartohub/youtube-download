#!/bin/bash
"""
Setup script for YouTube Downloader with Translation
This script creates a virtual environment and installs all required dependencies.
"""

echo "🚀 Setting up YouTube Downloader with Translation..."
echo "=================================================="

# Check if we're in the correct directory
if [ ! -f "youtube_gradio_app.py" ]; then
    echo "❌ Error: Please run this script from the youtube-srt directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created successfully"
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
echo "This may take a few minutes for the first time setup..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install requirements"
    exit 1
fi

echo ""

# Download NLTK punkt tokenizer
echo "📚 Downloading NLTK punkt tokenizer..."
python -c "import nltk; nltk.download('punkt', download_dir='nltk_data')"

echo "✅ Setup completed successfully!"
echo "=================================================="
echo ""
echo "🎯 To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the Gradio app: python youtube_gradio_app.py"
echo ""
echo "🌐 The web interface will open automatically in your browser"
echo "📱 Access it at: http://localhost:7860"
echo ""
echo "📝 Features:"
echo "  • Download YouTube videos and subtitles"
echo "  • Automatic English to Traditional Chinese translation (NLLB-200 3.3B)"
echo "  • Web-based user interface"
echo "  • Support for multiple subtitle languages"
echo "  • Advanced SRT manipulation and post-translation correction"
echo ""
