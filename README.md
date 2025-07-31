
# YouTube Video Downloader & Translator (Web & CLI)

---
A comprehensive Python toolkit and web app for downloading YouTube videos, extracting subtitles, and translating English subtitles to Traditional Chinese using local AI models. Includes both a command-line interface and a Gradio-based web interface.


## 🆕 Post-Translation Correction (Web & CLI)

**New Feature:** After subtitle translation, you can now review and correct the translated SRT before saving or downloading:

- **Web App:** The translated SRT appears in an editable text area. Make corrections, then click "Download Corrected Translation" to save your changes.
- **CLI:** After translation, the SRT file opens in your default text editor (set by `$EDITOR`, or falls back to `nano`, `vi`, or Notepad). Edit and save before closing the editor.

This ensures you can fix any translation mistakes before using the subtitles.

---

## ✨ Features

- 🎬 **Video Download**: Download YouTube videos in the highest quality (MP4/H.264, stream merging, or single-file mode)
- 📝 **Subtitle Download**: Extract subtitles in multiple languages (manual and auto-generated)
- ⚡ **Subtitle-Only Mode**: Download only subtitles (faster, smaller files)
- 🌏 **Local Translation**: Translate English subtitles to Traditional Chinese using local AI models (Helsinki-NLP/opus-mt-en-zh + OpenCC)
- 🖥️ **Web Interface**: Easy-to-use Gradio web app
- 🖥️ **Command-Line Interface**: Full-featured CLI for advanced users
- 📱 **Cross-platform**: Works on Windows, macOS, and Linux
- 🔒 **Privacy**: All processing done locally, no data sent to external services

---

## Quick Start (Web App)

1. **Setup** (one-time):
   ```bash
   ./setup.sh
   ```
2. **Run the application**:
   ```bash
   ./run.sh
   ```
3. **Open your browser** and go to `http://localhost:7860`

---

## Manual Setup

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the web app**:
   ```bash
   python youtube_gradio_app.py
   ```

---

## Usage

### Web Interface
1. **Enter YouTube URL**
2. **Get Video Info**
3. **Configure Options** (subtitle language, auto-captions, translation, quality)
4. **Choose Download Type** (Video & Subtitles, Subtitles Only)
5. **Review and Edit Translated Subtitles** in the editable area (optional)
6. **Download Files** (including corrected translation) from the web interface

### Command-Line Interface

#### Default Downloads (Video + Transcript)
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

**After translation:**
- If you download a translated SRT, it will automatically open in your default text editor for review and correction. Save and close the editor to finish.

#### Video-Only, Transcript-Only, Audio-Only, and Advanced Options
See below for more CLI examples and options.

---

## File Structure
```
youtube-srt/
├── youtube_gradio_app.py    # Main Gradio application
├── youtube_srt.py           # Command-line script
├── setup.sh                 # Setup script
├── run.sh                   # Run script
├── requirements.txt         # Python dependencies
├── downloads/               # Downloaded files folder
├── translation_models/      # Local translation model cache
└── venv/                    # Virtual environment
```

---

## Translation Model
- Uses Helsinki-NLP's English→Chinese model (`Helsinki-NLP/opus-mt-en-zh`)
- Runs completely offline after initial download (~300MB)
- Converts output to Traditional Chinese using OpenCC (if installed)
- Supports GPU acceleration if available

**Note:** If the translation model is not present, the app will automatically create the `translation_models` directory and download the required model on first use. No manual intervention is needed.

---

## System Requirements
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended for translation)
- **Storage**: 2GB free space for models and dependencies
- **Network**: Internet connection for initial setup and YouTube downloads

---

## Supported Formats
- **Video**: MP4 (H.264)
- **Subtitles**: SRT format
- **Languages**: All languages supported by YouTube subtitles
- **Translation**: English → Traditional Chinese

---

## Command-Line Usage & Options

### Default Downloads (Video + Transcript)
```bash
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
```
### Video-Only Downloads
```bash
python youtube_downloader.py --video-only "https://www.youtube.com/watch?v=VIDEO_ID"
```
### Transcript-Only Downloads
```bash
python youtube_downloader.py --transcript-only "https://www.youtube.com/watch?v=VIDEO_ID"
```
### Audio-Only Downloads
```bash
python youtube_downloader.py --audio-only "https://www.youtube.com/watch?v=VIDEO_ID"
```
### Advanced Examples
```bash
python youtube_downloader.py --include-auto-captions "https://www.youtube.com/watch?v=VIDEO_ID"
python youtube_downloader.py --list-transcripts "https://www.youtube.com/watch?v=VIDEO_ID"
```

#### CLI Options (Summary)
- `url`: YouTube video URL (required)
- `-o, --output`: Output directory (default: ./downloads)
- `--video-only`, `--transcript-only`, `--audio-only`: Download modes
- `--transcript-lang`: Language code for transcript (default: en)
- `--include-auto-captions`: Include auto-generated captions
- `--list-transcripts`: List all available transcripts/subtitles
- `--list-formats`: List all available video formats
- `--single-file`: Download single file with audio (faster, lower quality)
- `--force`: Force download even if file exists
- `-h, --help`: Show help message

---

## Troubleshooting & Tips

### Common Issues
- **"Virtual environment not found"**: Run `./setup.sh` first
- **"Required packages not installed"**: Run `./setup.sh` again
- **Translation model download fails**: Check internet connection, restart app
- **Out of memory during translation**: Close other apps, restart, or use more RAM
- **"yt-dlp is not installed" error**: Run `pip install yt-dlp`
- **"Video unavailable" error**: Try updating yt-dlp: `pip install --upgrade yt-dlp`
- **ffmpeg errors**: Install ffmpeg or use `--single-file` mode
- **No subtitles found**: Use `--list-transcripts` to see available languages

### Performance Tips
- **GPU Acceleration**: Used automatically if CUDA is available
- **Memory Usage**: Translation models require significant memory
- **Download Speed**: Depends on your internet connection and YouTube's servers

---

## Legal Notice
Please respect YouTube's Terms of Service and copyright laws. Only download videos and transcripts that you have permission to download or that are in the public domain.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing
Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📦 Installation

1. Make sure you have Python 3.6+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install yt-dlp directly:

```bash
pip install yt-dlp
```

3. **Required for video merging**: Install ffmpeg for stream merging:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from https://ffmpeg.org/download.html

*Note: ffmpeg is not required for transcript downloads or single-file video downloads.*

## 🚀 Usage

### Default Downloads (Video + Transcript)

```bash
# Download both video and transcript (default behavior)
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Download to specific directory
python youtube_downloader.py -o ./my_videos "https://www.youtube.com/watch?v=VIDEO_ID"

# Download with specific transcript language
python youtube_downloader.py --transcript-lang es "https://www.youtube.com/watch?v=VIDEO_ID"

# Disable auto-language selection (prompt user if language not found)
python youtube_downloader.py --no-auto-lang "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Video-Only Downloads

```bash
# Download video only (no transcript)
python youtube_downloader.py --video-only "https://www.youtube.com/watch?v=VIDEO_ID"

# Single file mode (faster, lower quality, no ffmpeg needed)
python youtube_downloader.py --video-only --single-file "https://www.youtube.com/watch?v=VIDEO_ID"

# Force download even if file exists
python youtube_downloader.py --video-only --force "https://www.youtube.com/watch?v=VIDEO_ID"

# List available video formats
python youtube_downloader.py --video-only --list-formats "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Transcript-Only Downloads

```bash
# Download transcript only (no video)
python youtube_downloader.py --transcript-only "https://www.youtube.com/watch?v=VIDEO_ID"

# Download Spanish transcript only
python youtube_downloader.py --transcript-only --transcript-lang es "https://www.youtube.com/watch?v=VIDEO_ID"

# Include auto-generated captions
python youtube_downloader.py --transcript-only --include-auto-captions "https://www.youtube.com/watch?v=VIDEO_ID"

# List all available transcript languages
python youtube_downloader.py --list-transcripts "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Audio-Only Downloads

```bash
# Download audio only (highest quality)
python youtube_downloader.py --audio-only "https://www.youtube.com/watch?v=VIDEO_ID"

# Download audio to specific directory
python youtube_downloader.py --audio-only -o ./my_audios "https://www.youtube.com/watch?v=VIDEO_ID"

# Force download audio even if file exists
python youtube_downloader.py --audio-only --force "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Examples

```bash
# Download transcript with auto-captions included
python youtube_downloader.py --include-auto-captions "https://www.youtube.com/watch?v=VIDEO_ID"

# Debug video formats and available transcripts
python youtube_downloader.py --debug "https://www.youtube.com/watch?v=VIDEO_ID"
python youtube_downloader.py --list-transcripts "https://www.youtube.com/watch?v=VIDEO_ID"
```

## 📋 Command Line Options

### General Options
- `url`: YouTube video URL (required)
- `-o, --output`: Output directory (default: ./downloads)
- `-h, --help`: Show help message

### Video Download Options
- `--video-only`: Download video only (instead of both video and transcript)
- `-f, --list-formats`: List all available video formats without downloading
- `-s, --single-file`: Download single file with audio (faster but lower quality)
- `-d, --debug`: Show detailed format information for debugging
- `--force`: Force download even if file exists

### Transcript Download Options
- `--transcript-only`: Download transcript/subtitles only (instead of both video and transcript)
- `--transcript-lang`: Language code for transcript (default: en)
- `--include-auto-captions`: Include auto-generated captions (default: manual only)
- `--no-auto-lang`: Disable automatic language selection (prompt user when language not found)
- `--list-transcripts`: List all available transcripts/subtitles for the video

### Audio Download Options
- `-a, --audio-only`: Download audio only in MP3 format (highest quality)

## 🎯 Video Quality Modes

### Default Mode (Stream Merging) - **Recommended**
Downloads the highest quality video stream and audio stream separately, then merges them using ffmpeg with stream copy (no re-encoding). This provides the best quality available (4K, 8K if available) while being fast since no re-encoding occurs.

### Single File Mode (--single-file flag)
Downloads the highest quality single file that already contains both video and audio. This is faster and simpler with no post-processing required, but quality is limited to what YouTube provides as pre-merged files (usually max 720p-1080p).

## 📝 Transcript Features

### Language Support
The script supports downloading transcripts in any language available for the video. Common language codes include:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `zh-Hans` - Chinese (Simplified)
- `zh-Hant` - Chinese (Traditional)
- `ja` - Japanese
- `ko` - Korean
- `pt` - Portuguese
- `ru` - Russian

### Transcript Types
1. **Manual Subtitles** (preferred): Human-created, high accuracy
2. **Auto-generated Captions**: Machine-generated, may contain errors

### Smart Language Handling
- **Manual Priority**: Always prefers manual subtitles over auto-generated
- **Interactive Fallback**: Prompts user when requested language unavailable
- **Auto-selection**: `--auto-lang` flag automatically uses first available language
- **Clear Feedback**: Shows what languages are available for each video

## 📁 Output Files

### Video Files
Downloaded videos are saved as: `{Video Title}.{Extension}`

### Transcript Files
Downloaded transcripts are saved as: `{Video Title}.{Language Code}.srt`

Example: `My Video.en.srt` or `我的视频.zh-Hans.srt`

### Audio Files
Downloaded audio files are saved as: `{Video Title}.mp3`

## 🛠 Troubleshooting

### Video Download Issues

1. **"yt-dlp is not installed" error**
   - Solution: Run `pip install yt-dlp`

2. **"Video unavailable" error**
   - The video might be private, deleted, or region-restricted
   - Try updating yt-dlp: `pip install --upgrade yt-dlp`

3. **ffmpeg errors in merge mode**
   - Install ffmpeg (see installation section)
   - Use `--single-file` flag as alternative

### Transcript Download Issues

1. **"No subtitles found for language 'en'" error**
   - Video may not have English transcripts
   - Use `--list-transcripts` to see available languages
   - Use `--auto-lang` to automatically select available language
   - Try `--include-auto-captions` if only auto-captions available

2. **Interactive prompt doesn't work in scripts**
   - Use `--auto-lang` flag for non-interactive environments
   - Specify exact language with `--transcript-lang`

3. **Transcript file not found**
   - Check the downloads directory
   - Video may not have any transcripts available
   - Some videos only have auto-generated captions

### Common Solutions

```bash
# Check what's available before downloading
python youtube_downloader.py --list-transcripts "URL"
python youtube_downloader.py --list-formats "URL"

# Download with maximum flexibility
python youtube_downloader.py --transcript --auto-lang --include-auto-captions "URL"

# Fallback for difficult videos
python youtube_downloader.py --single-file "URL"
```

## 📄 Legal Notice

Please respect YouTube's Terms of Service and copyright laws. Only download videos and transcripts that you have permission to download or that are in the public domain.

## 🔄 Updates

To update yt-dlp to the latest version:

```bash
pip install --upgrade yt-dlp
```

This ensures compatibility with the latest YouTube changes.

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
