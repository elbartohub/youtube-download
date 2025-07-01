# YouTube Video Downloader

A Python script to download YouTube videos in the highest quality available using `yt-dlp`.

## Features

- Downloads YouTube videos in the highest quality by default using stream merging
- Uses ffmpeg stream copy (no re-encoding) for fast, lossless merging
- Prefers H.264 codec for maximum compatibility (QuickTime, etc.)
- Shows video information before downloading
- Progress indicator during download
- Error handling and user-friendly messages
- Supports various YouTube URL formats
- Option to list available video formats
- Option for single-file downloads (faster but lower quality)
- Customizable output directory

## Installation

1. Make sure you have Python 3.6+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install yt-dlp directly:

```bash
pip install yt-dlp
```

3. **Required for default mode**: Install ffmpeg for stream merging:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: Download from https://ffmpeg.org/download.html

## Usage

### Basic Usage

```bash
./youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Advanced Usage

```bash
# Download to specific directory
./youtube_downloader.py -o ./my_videos "https://www.youtube.com/watch?v=VIDEO_ID"

# Force download even if file exists
./youtube_downloader.py --force "https://www.youtube.com/watch?v=VIDEO_ID"

# Download with single file mode (faster but lower quality)
./youtube_downloader.py -s "https://www.youtube.com/watch?v=VIDEO_ID"

# List available formats without downloading
./youtube_downloader.py -f "https://www.youtube.com/watch?v=VIDEO_ID"

# Show help
./youtube_downloader.py -h
```

### Examples

```bash
# Download a video to default ./downloads folder
./youtube_downloader.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Download to custom folder
./youtube_downloader.py --output ./music_videos "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Force download to create duplicate
./youtube_downloader.py --force "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Download with single file mode (faster, lower quality)
./youtube_downloader.py --single-file "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Check available formats first
./youtube_downloader.py --list-formats "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

## Command Line Options

- `url`: YouTube video URL (required)
- `-o, --output`: Output directory for downloaded videos (default: ./downloads)
- `-f, --list-formats`: List all available formats without downloading
- `-s, --single-file`: Download single file with audio (faster but lower quality)
- `-d, --debug`: Show detailed format information for debugging
- `--force`: Force download even if file exists (overwrites existing file)
- `-h, --help`: Show help message

## File Handling

- **Default behavior**: If a file already exists, the download is skipped
- **Force mode** (`--force`): Downloads anyway and overwrites the existing file

## Video Quality

The script has two download modes:

### Default Mode (Stream Merging) - **Recommended**
Downloads the highest quality video stream and audio stream separately, then merges them using ffmpeg with stream copy (no re-encoding). This provides the best quality available (4K, 8K if available) while being fast since no re-encoding occurs.

### Single File Mode (--single-file flag)
Downloads the highest quality single file that already contains both video and audio. This is faster and simpler with no post-processing required, but quality is limited to what YouTube provides as pre-merged files (usually max 720p-1080p).

**Note**: ffmpeg is required for the default merge mode but provides much better quality.

To modify the quality settings:

1. Edit the `ydl_opts` in the `YouTubeDownloader` class
2. For single file mode: `'best[ext=mp4]/best'`
3. For merge mode: `'bestvideo+bestaudio/best'`
4. Other options:
   - `'best[height<=1080]'`: Best quality up to 1080p
   - `'best[height<=720]'`: Best quality up to 720p
   - `'worst'`: Lowest quality available

## Output

Downloaded videos are saved with the format: `{Video Title}.{Extension}`

## Error Handling

The script handles common errors such as:
- Invalid URLs
- Network connectivity issues
- Video not available
- Private or restricted videos

## Requirements

- Python 3.6+
- yt-dlp library

## Legal Notice

Please respect YouTube's Terms of Service and copyright laws. Only download videos that you have permission to download or that are in the public domain.

## Troubleshooting

### Common Issues

1. **"yt-dlp is not installed" error**
   - Solution: Run `pip install yt-dlp`

2. **"Video unavailable" error**
   - The video might be private, deleted, or region-restricted
   - Try updating yt-dlp: `pip install --upgrade yt-dlp`

3. **Slow download speeds**
   - This depends on your internet connection and YouTube's servers
   - Try downloading during off-peak hours

4. **Permission errors**
   - Make sure you have write permissions to the output directory
   - Try running with elevated permissions if necessary

## Updates

To update yt-dlp to the latest version:

```bash
pip install --upgrade yt-dlp
```

This ensures compatibility with the latest YouTube changes.
