#!/usr/bin/env python3
"""
YouTube Video Downloader Script

This script allows users to download YouTube videos in the highest quality available.
It uses yt-dlp library for downloading and supports various video formats.

Usage:
    python youtube_downloader.py <YouTube_URL>
    python youtube_downloader.py -h  # for help

Features:
- Downloads in highest quality available
- Shows download progress
- Handles errors gracefully
- Supports various YouTube URL formats
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed.")
    print("Please install it using: pip install yt-dlp")
    sys.exit(1)


class YouTubeDownloader:
    def __init__(self, download_path="./downloads", merge_streams=True, force_download=False):
        """Initialize the YouTube downloader."""
        self.download_path = Path(download_path)
        self.download_path.mkdir(exist_ok=True)
        self.merge_streams = merge_streams
        
        # Base yt-dlp options
        self.base_opts = {
            'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),  
            'noplaylist': True,  # Download single video even if URL contains playlist
            'nooverwrites': not force_download,  # Allow overwrite if force is enabled
        }
        
        # yt-dlp options for highest quality
        if merge_streams:
            # Download separate video and audio streams, then merge WITHOUT re-encoding
            # Prefer H.264 (avc1) codec for better compatibility with QuickTime and other players
            self.ydl_opts = {
                **self.base_opts,
                'format': 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
                'merge_output_format': 'mp4',  # Ensure final output is MP4
                'postprocessor_args': {
                    'ffmpeg': ['-c', 'copy']  # Use stream copy instead of re-encoding
                },
                'prefer_ffmpeg': True,  # Prefer ffmpeg over avconv
            }
        else:
            # We'll set the format dynamically after analyzing available formats
            self.ydl_opts = {
                **self.base_opts,
                'format': 'best',  # Default fallback
            }
    
    def download_video(self, url):
        """Download a single video from YouTube."""
        try:
            # First, get video info to analyze available formats
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                print(f"Fetching video information...")
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Unknown')
                
                # For single file mode, intelligently select the best format
                if not self.merge_streams:
                    formats = info.get('formats', [])
                    best_format_id = self._select_best_single_format(formats)
                    
                    if best_format_id:
                        # Update the format selector with the specific format ID
                        self.ydl_opts['format'] = best_format_id
                    else:
                        print("⚠️  No single file formats with audio+video found, using default selection")
                
                print(f"\nVideo Details:")
                print(f"Title: {title}")
                print(f"Uploader: {uploader}")
                print(f"Duration: {self._format_duration(duration)}")
                print(f"Download mode: {'Merge streams (highest quality, stream copy)' if self.merge_streams else 'Single file with audio (lower quality, faster)'}")
                
                # Download the video with the updated format selection
                print(f"\n🚀 Starting download: {title}")
                print(f"📁 Saving to: {self.download_path}")
                if self.base_opts['nooverwrites']:
                    print(f"📝 Note: If file exists, it will be skipped (use --force to download anyway)")
                else:
                    print(f"📝 Note: Force mode - will create numbered duplicate if file exists")
                
            # Now download with the selected format
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
                
                print(f"\n✅ Successfully downloaded: {title}")
                return True
                
        except yt_dlp.DownloadError as e:
            print(f"❌ Download error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def _format_duration(self, seconds):
        """Format duration from seconds to readable format."""
        if not seconds:
            return "Unknown"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_video_formats(self, url):
        """List available formats for a video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                print(f"\nAvailable formats for: {info.get('title', 'Unknown')}")
                print("-" * 60)
                
                for f in formats:
                    format_id = f.get('format_id', 'N/A')
                    ext = f.get('ext', 'N/A')
                    resolution = f.get('resolution', 'N/A')
                    filesize = f.get('filesize', 0)
                    
                    size_str = f"{filesize // (1024*1024)}MB" if filesize else "Unknown size"
                    print(f"ID: {format_id:10} | {ext:4} | {resolution:10} | {size_str}")
                
        except Exception as e:
            print(f"Error getting formats: {e}")
    
    def show_format_debug(self, url):
        """Show detailed format information for debugging."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                print(f"\n🔍 FORMAT DEBUG for: {info.get('title', 'Unknown')}")
                print("=" * 60)
                
                # Show single files with both audio and video
                print("\n📹 Single files with audio and video:")
                single_files = []
                for f in formats:
                    acodec = f.get('acodec', 'none')
                    vcodec = f.get('vcodec', 'none')
                    if acodec != 'none' and vcodec != 'none':
                        height = f.get('height', 0)
                        ext = f.get('ext', 'unknown')
                        filesize = f.get('filesize', 0)
                        size_str = f"{filesize // (1024*1024)}MB" if filesize else "Unknown"
                        single_files.append((height, f))
                        print(f"  {height}p | {ext} | {size_str} | ID: {f.get('format_id', 'N/A')}")
                
                # Show the highest resolution single file
                if single_files:
                    single_files.sort(key=lambda x: x[0], reverse=True)
                    highest_single = single_files[0][1]
                    print(f"\n🏆 Highest resolution single file: {highest_single.get('height', 'Unknown')}p")
                else:
                    print("\n❌ No single files with both audio and video found!")
                
                # Show separate video streams
                print(f"\n🎬 Separate video streams (no audio):")
                video_only = [f for f in formats if f.get('acodec') == 'none' and f.get('vcodec') != 'none']
                for f in sorted(video_only, key=lambda x: x.get('height', 0), reverse=True)[:5]:
                    height = f.get('height', 0)
                    ext = f.get('ext', 'unknown')
                    print(f"  {height}p | {ext} | ID: {f.get('format_id', 'N/A')}")
                
        except Exception as e:
            print(f"Debug error: {e}")
    
    def _select_best_single_format(self, formats):
        """Select the best single format that contains both audio and video."""
        # Filter formats that have both audio and video
        single_formats = []
        video_only_formats = []
        
        for f in formats:
            acodec = f.get('acodec', 'none')
            vcodec = f.get('vcodec', 'none')
            height = f.get('height', 0)
            
            # Must have both audio and video, and a reasonable resolution
            if acodec != 'none' and vcodec != 'none' and height > 0:
                single_formats.append(f)
            # Track video-only streams for comparison
            elif acodec == 'none' and vcodec != 'none' and height > 0:
                video_only_formats.append(f)
        
        if not single_formats:
            return None
        
        # Sort by resolution (height) in descending order
        single_formats.sort(key=lambda x: x.get('height', 0), reverse=True)
        video_only_formats.sort(key=lambda x: x.get('height', 0), reverse=True)
        
        # Get the best available formats
        best_single = single_formats[0]
        best_single_res = best_single.get('height', 0)
        
        # Check if there are much higher resolution video-only streams
        if video_only_formats:
            best_video_only_res = video_only_formats[0].get('height', 0)
            
            # If video-only is significantly higher resolution, suggest merge mode
            if best_video_only_res > best_single_res * 2:  # More than double the resolution
                print(f"📊 Selected single file format: {best_single_res}p ({best_single.get('ext', 'unknown')})")
                print(f"⚠️  Note: Higher quality available ({best_video_only_res}p) in default merge mode!")
                print(f"💡 Remove --single-file flag for {best_video_only_res}p quality")
            else:
                print(f"📊 Selected format: {best_single_res}p ({best_single.get('ext', 'unknown')}) - ID: {best_single.get('format_id', 'N/A')}")
        else:
            print(f"📊 Selected format: {best_single_res}p ({best_single.get('ext', 'unknown')}) - ID: {best_single.get('format_id', 'N/A')}")
        
        return best_single.get('format_id')


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube videos in highest quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Download highest quality (merge mode)
  %(prog)s --list-formats "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --output ./my_videos "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --single-file "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Single file mode
        """
    )
    
    parser.add_argument(
        'url',
        help='YouTube video URL to download'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='./downloads',
        help='Output directory for downloaded videos (default: ./downloads)'
    )
    
    parser.add_argument(
        '-f', '--list-formats',
        action='store_true',
        help='List all available formats for the video without downloading'
    )
    
    parser.add_argument(
        '-s', '--single-file',
        action='store_true',
        help='Download single file with audio (lower quality but faster, no ffmpeg needed)'
    )
    
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Show detailed format information for debugging'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force download even if file exists (will create numbered duplicate)'
    )
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url:
        print("Error: Please provide a YouTube URL")
        parser.print_help()
        sys.exit(1)
    
    # Check if URL looks like a YouTube URL
    youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
    if not any(domain in args.url.lower() for domain in youtube_domains):
        print("Warning: The provided URL doesn't appear to be a YouTube URL")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes']:
            sys.exit(1)
    
    # Initialize downloader (merge is default, single-file is the option)
    downloader = YouTubeDownloader(args.output, merge_streams=not args.single_file, force_download=args.force)
    
    # List formats, debug, or download
    if args.list_formats:
        downloader.get_video_formats(args.url)
    elif args.debug:
        downloader.show_format_debug(args.url)
    else:
        success = downloader.download_video(args.url)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
