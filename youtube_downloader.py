#!/usr/bin/env python3
"""
YouTube Video Downloader Script

This script allows users to download YouTube videos in the highest quality available
and download transcripts/subtitles as SRT files. By default, it downloads both video
and transcript with automatic language selection.
It uses yt-dlp library for downloading and supports various video formats.

Usage:
    python youtube_downloader.py <YouTube_URL>  # Downloads both video + transcript (default)
    python youtube_downloader.py --video-only <YouTube_URL>  # Video only
    python youtube_downloader.py --transcript-only <YouTube_URL>  # Transcript only
    python youtube_downloader.py -h  # for help

Features:
- Downloads both video and transcript by default with auto-language selection
- Downloads in highest quality available
- Downloads transcripts/subtitles as SRT files (manual subtitles by default)
- Combined download mode for video + transcript in one command
- Shows download progress
- Handles errors gracefully
- Supports various YouTube URL formats
- Language selection for transcripts
- Optional auto-generated captions support
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
    
    def download_both(self, url, transcript_lang='en', auto_captions=False, auto_lang=False):
        """Download both video and transcript in one operation."""
        print("🎬📝 Starting combined download: Video + Transcript")
        print("=" * 60)
        
        # Download video first
        print("\n1️⃣ DOWNLOADING VIDEO")
        print("-" * 30)
        video_success = self.download_video(url)
        
        # Download transcript second
        print("\n2️⃣ DOWNLOADING TRANSCRIPT")
        print("-" * 30)
        transcript_success = self.download_transcript(url, transcript_lang, auto_captions, auto_lang)
        
        # Summary
        print("\n📊 DOWNLOAD SUMMARY")
        print("-" * 30)
        if video_success and transcript_success:
            print("✅ Video: Downloaded successfully")
            print("✅ Transcript: Downloaded successfully")
            print("🎉 Both downloads completed successfully!")
            return True
        elif video_success and not transcript_success:
            print("✅ Video: Downloaded successfully")
            print("❌ Transcript: Failed to download")
            print("⚠️ Video downloaded, but transcript failed")
            return False
        elif not video_success and transcript_success:
            print("❌ Video: Failed to download")
            print("✅ Transcript: Downloaded successfully")
            print("⚠️ Transcript downloaded, but video failed")
            return False
        else:
            print("❌ Video: Failed to download")
            print("❌ Transcript: Failed to download")
            print("💥 Both downloads failed")
            return False

    def download_transcript(self, url, language='en', auto_captions=False, auto_lang=False):
        """Download transcript/subtitles as SRT file."""
        try:
            # Configure yt-dlp options for transcript download
            transcript_opts = {
                'writesubtitles': True,
                'writeautomaticsub': auto_captions,  # Include auto-generated captions
                'subtitleslangs': [language],
                'subtitlesformat': 'srt',
                'skip_download': True,  # Only download subtitles, not video
                'outtmpl': str(self.download_path / '%(title)s.%(ext)s'),
            }
            
            # First, get video info to check available subtitles
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                print(f"Fetching video information...")
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', 'Unknown')
                uploader = info.get('uploader', 'Unknown')
                
                # Check available subtitles
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                print(f"\nVideo: {title}")
                print(f"Uploader: {uploader}")
                
                # Show available subtitle languages
                if subtitles:
                    print(f"\n📝 Manual subtitles available: {', '.join(subtitles.keys())}")
                if automatic_captions:
                    print(f"🤖 Auto-generated captions available: {', '.join(automatic_captions.keys())}")
                
                # Check if requested language is available
                has_manual = language in subtitles
                has_auto = language in automatic_captions
                
                if not has_manual and not has_auto:
                    print(f"❌ No subtitles found for language '{language}'")
                    if subtitles or automatic_captions:
                        available_langs = list(set(list(subtitles.keys()) + list(automatic_captions.keys())))
                        print(f"💡 Available languages: {', '.join(available_langs)}")
                        
                        # Offer to download the first available language
                        if available_langs:
                            first_lang = available_langs[0]
                            if auto_lang:
                                print(f"🔄 Auto-switching to available language: {first_lang}")
                                language = first_lang
                                transcript_opts['subtitleslangs'] = [language]
                                
                                # Update availability checks
                                has_manual = language in subtitles
                                has_auto = language in automatic_captions
                            else:
                                print(f"🤔 Would you like to download '{first_lang}' instead?")
                                try:
                                    choice = input("Enter 'y' to download, or any other key to cancel: ").lower().strip()
                                    if choice in ['y', 'yes']:
                                        print(f"📥 Switching to language: {first_lang}")
                                        # Update the language and transcript options
                                        language = first_lang
                                        transcript_opts['subtitleslangs'] = [language]
                                        
                                        # Update availability checks
                                        has_manual = language in subtitles
                                        has_auto = language in automatic_captions
                                    else:
                                        return False
                                except (EOFError, KeyboardInterrupt):
                                    print("\n❌ Download cancelled")
                                    return False
                        else:
                            return False
                    else:
                        return False
                
                # Prioritize manual subtitles over auto-captions
                if has_manual:
                    print(f"✅ Found manual subtitles for '{language}'")
                elif has_auto and auto_captions:
                    print(f"🤖 Found auto-generated captions for '{language}'")
                elif has_auto and not auto_captions:
                    print(f"❌ Only auto-generated captions available for '{language}', but auto-captions are disabled")
                    print(f"💡 Use --include-auto-captions to download auto-generated captions")
                    return False
                
            # Download the transcript
            print(f"\n🚀 Downloading transcript...")
            print(f"📁 Saving to: {self.download_path}")
            print(f"📝 Mode: {'Manual subtitles + auto-captions' if auto_captions else 'Manual subtitles only'}")
            
            with yt_dlp.YoutubeDL(transcript_opts) as ydl:
                ydl.download([url])
                
                # Find the downloaded SRT file
                expected_filename = f"{title}.{language}.srt"
                srt_path = self.download_path / expected_filename
                
                if srt_path.exists():
                    print(f"✅ Transcript downloaded: {expected_filename}")
                else:
                    # Look for any SRT files in the directory that might match
                    srt_files = list(self.download_path.glob("*.srt"))
                    if srt_files:
                        latest_srt = max(srt_files, key=os.path.getctime)
                        print(f"✅ Transcript downloaded: {latest_srt.name}")
                    else:
                        print("⚠️ Transcript download completed but file not found")
                
                return True
                
        except yt_dlp.DownloadError as e:
            print(f"❌ Transcript download error: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
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
    
    def list_available_transcripts(self, url):
        """List all available transcripts/subtitles for a video."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', 'Unknown')
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                print(f"\nAvailable transcripts for: {title}")
                print("-" * 60)
                
                if subtitles:
                    print("\n📝 Manual Subtitles:")
                    for lang, formats in subtitles.items():
                        format_list = [f.get('ext', 'unknown') for f in formats]
                        print(f"  {lang}: {', '.join(format_list)}")
                
                if automatic_captions:
                    print("\n🤖 Auto-generated Captions:")
                    for lang, formats in automatic_captions.items():
                        format_list = [f.get('ext', 'unknown') for f in formats]
                        print(f"  {lang}: {', '.join(format_list)}")
                
                if not subtitles and not automatic_captions:
                    print("\n❌ No subtitles or captions available for this video")
                
                return True
                
        except Exception as e:
            print(f"Error listing transcripts: {e}")
            return False
    
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
        description="Download YouTube videos and transcripts (both by default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Download video + transcript (default)
  %(prog)s --video-only "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Download video only
  %(prog)s --transcript-only "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Download transcript only
  %(prog)s --transcript-lang es "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Spanish transcript + video
  %(prog)s --no-auto-lang "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Disable auto-language (prompt user)
  %(prog)s --list-formats "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --output ./my_videos "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  %(prog)s --single-file "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Single file mode
  %(prog)s --include-auto-captions "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Include auto-captions
  %(prog)s --list-transcripts "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # List available transcripts
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
    
    parser.add_argument(
        '-t', '--transcript-only',
        action='store_true',
        help='Download transcript/subtitles only (instead of both video and transcript)'
    )
    
    parser.add_argument(
        '-v', '--video-only',
        action='store_true',
        help='Download video only (instead of both video and transcript)'
    )
    
    parser.add_argument(
        '--transcript-lang',
        default='en',
        help='Language code for transcript (default: en)'
    )
    
    parser.add_argument(
        '--include-auto-captions',
        action='store_true',
        help='Include auto-generated captions (default: manual subtitles only)'
    )
    
    parser.add_argument(
        '--no-auto-lang',
        action='store_true',
        help='Disable automatic language selection (prompt user when language not found) - Default: auto-select available language'
    )
    
    parser.add_argument(
        '--list-transcripts',
        action='store_true',
        help='List all available transcripts/subtitles for the video'
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
    
    # Handle different modes
    if args.list_formats:
        downloader.get_video_formats(args.url)
    elif args.list_transcripts:
        downloader.list_available_transcripts(args.url)
    elif args.debug:
        downloader.show_format_debug(args.url)
    elif args.transcript_only:
        # Download transcript only
        auto_captions = args.include_auto_captions
        auto_lang = not args.no_auto_lang  # auto_lang is True by default unless --no-auto-lang is specified
        success = downloader.download_transcript(args.url, args.transcript_lang, auto_captions, auto_lang)
        sys.exit(0 if success else 1)
    elif args.video_only:
        # Download video only
        success = downloader.download_video(args.url)
        sys.exit(0 if success else 1)
    else:
        # Default: Download both video and transcript (default behavior)
        auto_captions = args.include_auto_captions
        auto_lang = not args.no_auto_lang  # auto_lang is True by default unless --no-auto-lang is specified
        success = downloader.download_both(args.url, args.transcript_lang, auto_captions, auto_lang)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
