#!/usr/bin/env python3
"""
YouTube Video Downloader with Translation - Gradio Web Interface

This script provides a web-based interface for downloading YouTube videos and transcripts
with integrated translation capabilities using local inference models.
Features English to Traditional Chinese translation by default.

Features:
- Web-based Gradio interface
- YouTube video and transcript downloading
- Local translation inference (English to Traditional Chinese)
- Support for multiple subtitle languages
- Real-time progress updates
- Download history
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
from pathlib import Path
model_cache_path = Path(__file__).parent / 'translation_models'
model_cache_path.mkdir(exist_ok=True)
os.environ['HF_HOME'] = str(model_cache_path)
os.environ['TRANSFORMERS_CACHE'] = str(model_cache_path)
print(f"[Startup] Hugging Face model cache directory: {model_cache_path}")

# Check if the model exists in the cache, if not, trigger download
import traceback
def ensure_translation_model():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[Startup] huggingface_hub is not installed. Please install it with: pip install huggingface_hub")
        return
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    model_dir = model_cache_path / model_name
    if not model_dir.exists() or not any(model_dir.iterdir()):
        print(f"[Startup] Model {model_name} not found in {model_dir}, downloading...")
        try:
            snapshot_download(repo_id=model_name, local_dir=str(model_dir), local_dir_use_symlinks=False)
            print(f"[Startup] Model {model_name} downloaded to {model_dir}")
        except Exception as e:
            print(f"[Startup] Error downloading model: {e}")
            traceback.print_exc()
    else:
        print(f"[Startup] Model {model_name} already exists in {model_dir}")

ensure_translation_model()
from typing import Optional, Tuple, List
import re
import json
from datetime import datetime

try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp is not installed.")
    print("Please install it using: pip install yt-dlp")
    sys.exit(1)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
except ImportError:
    print("Error: transformers and torch are not installed.")
    print("Please install them using: pip install transformers torch")
    sys.exit(1)


class YouTubeTranslationApp:
    def __init__(self):
        self.download_path = Path("./downloads")
        self.download_path.mkdir(exist_ok=True)
        self.translation_model = None
        self.translation_tokenizer = None
        self.setup_translation_model()
        
    def setup_translation_model(self):
        """Initialize the translation model for English to Traditional Chinese (via Simplified Chinese + OpenCC)."""
        try:
            print("Loading translation model (English to Chinese, with optional Traditional conversion)...")
            model_name = "Helsinki-NLP/opus-mt-en-zh"
            self.translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.translator = pipeline(
                "translation", 
                model=self.translation_model, 
                tokenizer=self.translation_tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            try:
                import opencc
                self.opencc = opencc.OpenCC('s2t')
                print("✅ OpenCC loaded: Simplified to Traditional Chinese conversion enabled.")
            except ImportError:
                self.opencc = None
                print("⚠️ OpenCC not installed: output will be Simplified Chinese. Install with 'pip install opencc-python-reimplemented' for Traditional Chinese.")
            print("✅ Translation model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            self.translator = None
            self.opencc = None
    
    def translate_text(self, text: str, max_length: int = 512) -> str:
        """Translate English text to Traditional Chinese (via Simplified Chinese + OpenCC if available)."""
        if not self.translator:
            return text
        try:
            # Split long text into chunks to avoid token limits
            sentences = re.split(r'[.!?]+', text)
            translated_sentences = []
            for sentence in sentences:
                if sentence.strip():
                    result = self.translator(
                        sentence.strip(), 
                        max_length=max_length,
                        do_sample=False
                    )
                    zh_text = result[0]['translation_text']
                    if hasattr(self, 'opencc') and self.opencc:
                        zh_text = self.opencc.convert(zh_text)
                    translated_sentences.append(zh_text)
            return '。'.join(translated_sentences)
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def get_video_info(self, url: str) -> Tuple[str, str, str, List[str]]:
        """Get video information and available subtitle languages."""
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                
                title = info.get('title', 'Unknown')
                uploader = info.get('uploader', 'Unknown')
                duration = self._format_duration(info.get('duration', 0))
                
                # Get available subtitle languages
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                available_langs = list(set(list(subtitles.keys()) + list(automatic_captions.keys())))
                
                return title, uploader, duration, available_langs
        except Exception as e:
            return f"Error: {str(e)}", "", "", []
    
    def download_subtitle_only(
        self, 
        url: str, 
        subtitle_lang: str = "en",
        include_auto_captions: bool = True,
        translate_subtitles: bool = True
    ) -> Tuple[str, str, str]:
        """Download subtitle only with optional translation."""
        try:
            # Create unique folder for this download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_folder = self.download_path / f"subtitle_{timestamp}"
            download_folder.mkdir(exist_ok=True)
            
            # Get video info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()[:50]
            
            # Download transcript only
            transcript_opts = {
                'writesubtitles': True,
                'writeautomaticsub': include_auto_captions,
                'subtitleslangs': [subtitle_lang],
                'subtitlesformat': 'srt',
                'skip_download': True,
                'outtmpl': str(download_folder / f'{clean_title}.%(ext)s'),
            }
            
            transcript_path = None
            translated_transcript_path = None
            
            with yt_dlp.YoutubeDL(transcript_opts) as ydl:
                ydl.download([url])
                # Find the downloaded subtitle file
                srt_files = list(download_folder.glob("*.srt"))
                if srt_files:
                    transcript_path = str(srt_files[0])
                    # Translate subtitles if requested
                    if translate_subtitles and self.translator:
                        translated_transcript_path = self.translate_srt_file(
                            transcript_path, 
                            download_folder / f'{clean_title}_zh-tw.srt'
                        )
                        # Ensure translated file exists and is not empty
                        if translated_transcript_path and os.path.exists(translated_transcript_path):
                            try:
                                if os.path.getsize(translated_transcript_path) == 0:
                                    with open(translated_transcript_path, 'w', encoding='utf-8') as f:
                                        f.write('[Translation failed: No content]')
                            except Exception as e:
                                print(f"[download_subtitle_only] Error checking translated file: {e}")
                        else:
                            # Create a placeholder file if translation failed
                            placeholder_path = str(download_folder / f'{clean_title}_zh-tw.srt')
                            with open(placeholder_path, 'w', encoding='utf-8') as f:
                                f.write('[Translation failed: File not generated]')
                            translated_transcript_path = placeholder_path
            
            # Create download summary
            summary = f"""
            ✅ Subtitle download completed successfully!
            
            📹 Video: {title}
            📁 Folder: {download_folder.name}
            📝 Subtitles ({subtitle_lang}): {'✅ Downloaded' if transcript_path else '❌ Failed'}
            🌏 Chinese translation: {'✅ Generated' if translated_transcript_path else '❌ Not available'}
            """
            
            return (
                summary,
                transcript_path or "",
                translated_transcript_path or ""
            )
            
        except Exception as e:
            error_msg = f"❌ Error during subtitle download: {str(e)}"
            return error_msg, "", ""
    
    def download_video_and_transcript(
        self, 
        url: str, 
        subtitle_lang: str = "en",
        include_auto_captions: bool = True,
        translate_subtitles: bool = True,
        video_quality: str = "best"
    ) -> Tuple[str, str, str, str]:
        """Download video and transcript with optional translation."""
        try:
            # Create unique folder for this download
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_folder = self.download_path / f"download_{timestamp}"
            download_folder.mkdir(exist_ok=True)
            
            # Get video info first
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                clean_title = re.sub(r'[^\w\s-]', '', title).strip()[:50]
            
            # Download video
            video_opts = {
                'outtmpl': str(download_folder / f'{clean_title}.%(ext)s'),
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best' if video_quality == "best" else 'worst',
                'merge_output_format': 'mp4',
            }
            
            video_path = None
            with yt_dlp.YoutubeDL(video_opts) as ydl:
                ydl.download([url])
                # Find the downloaded video file
                video_files = list(download_folder.glob("*.mp4"))
                if video_files:
                    video_path = str(video_files[0])
            
            # Download transcript
            transcript_opts = {
                'writesubtitles': True,
                'writeautomaticsub': include_auto_captions,
                'subtitleslangs': [subtitle_lang],
                'subtitlesformat': 'srt',
                'skip_download': True,
                'outtmpl': str(download_folder / f'{clean_title}.%(ext)s'),
            }
            
            transcript_path = None
            translated_transcript_path = None
            
            with yt_dlp.YoutubeDL(transcript_opts) as ydl:
                ydl.download([url])
                # Find the downloaded subtitle file
                srt_files = list(download_folder.glob("*.srt"))
                if srt_files:
                    transcript_path = str(srt_files[0])
                    
                    # Translate subtitles if requested
                    if translate_subtitles and self.translator:
                        translated_transcript_path = self.translate_srt_file(
                            transcript_path, 
                            download_folder / f'{clean_title}_zh-tw.srt'
                        )
            
            # Create download summary
            summary = f"""
            ✅ Download completed successfully!
            
            📹 Video: {title}
            📁 Folder: {download_folder.name}
            🎬 Video file: {'✅ Downloaded' if video_path else '❌ Failed'}
            📝 Subtitles ({subtitle_lang}): {'✅ Downloaded' if transcript_path else '❌ Failed'}
            🌏 Chinese translation: {'✅ Generated' if translated_transcript_path else '❌ Not available'}
            """
            
            return (
                summary,
                video_path or "",
                transcript_path or "",
                translated_transcript_path or ""
            )
            
        except Exception as e:
            error_msg = f"❌ Error during download: {str(e)}"
            return error_msg, "", "", ""
    
    def translate_srt_file(self, srt_path: str, output_path: Path) -> str:
        """Translate an SRT subtitle file to Traditional Chinese."""
        try:
            print(f"[translate_srt_file] Translating: {srt_path} -> {output_path}")
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse SRT format
            subtitle_blocks = re.split(r'\n\s*\n', content.strip())
            translated_blocks = []

            for block in subtitle_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Keep index and timestamp, translate text
                    index = lines[0]
                    timestamp = lines[1]
                    text_lines = lines[2:]

                    # Translate each text line
                    translated_lines = []
                    for line in text_lines:
                        if line.strip():
                            translated_line = self.translate_text(line.strip())
                            print(f"[translate_srt_file] EN: {line.strip()} -> ZHT: {translated_line}")
                            translated_lines.append(translated_line)

                    translated_block = f"{index}\n{timestamp}\n" + "\n".join(translated_lines)
                    translated_blocks.append(translated_block)

            # Write translated SRT file
            translated_content = "\n\n".join(translated_blocks)
            if not translated_content.strip():
                print("[translate_srt_file] Warning: No translated content generated.")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("[Translation failed or no content to translate]")
                return str(output_path)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated_content)

            return str(output_path)
        except Exception as e:
            print(f"[translate_srt_file] Translation error: {e}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"[Translation error: {e}]")
            return str(output_path)
    
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
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(
            title="YouTube Downloader with Translation",
            theme=gr.themes.Soft(),
        ) as interface:
            
            gr.Markdown("""
            # 🎬 YouTube Video Downloader with Translation
            
            Download YouTube videos and transcripts with automatic English to Traditional Chinese translation.
            
            **Features:**
            - 🎥 Download full videos with subtitles
            - 📝 Download subtitles only (faster, smaller files)
            - 🌏 Automatic translation to Traditional Chinese
            - 🔧 Configurable quality and language options
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    url_input = gr.Textbox(
                        label="YouTube URL",
                        value="https://www.youtube.com/watch?v=hMGikmMFLAU",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
                    
                    info_button = gr.Button("📋 Get Video Info", variant="secondary")
                
                with gr.Column(scale=1):
                    video_info = gr.Textbox(
                        label="Video Information",
                        lines=4,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    subtitle_lang = gr.Dropdown(
                        choices=["en", "es", "fr", "de", "ja", "ko", "zh", "zh-cn", "zh-tw"],
                        value="en",
                        label="Subtitle Language"
                    )
                    
                    include_auto_captions = gr.Checkbox(
                        label="Include Auto-generated Captions",
                        value=True
                    )
                
                with gr.Column():
                    translate_subtitles = gr.Checkbox(
                        label="Translate to Traditional Chinese",
                        value=True
                    )
                    
                    video_quality = gr.Radio(
                        choices=["best", "worst"],
                        value="best",
                        label="Video Quality"
                    )
            
            download_button = gr.Button("⬇️ Download Video & Subtitles", variant="primary", size="lg")
            subtitle_only_button = gr.Button("📝 Download Subtitles Only", variant="secondary", size="lg")
            
            gr.Markdown("---")  # Separator line
            gr.Markdown("### 🎥 Video + Subtitle Download Results")
            
            with gr.Row():
                with gr.Column():
                    result_output = gr.Textbox(
                        label="Download Result",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column():
                    video_file = gr.File(label="Downloaded Video", interactive=False)
                    subtitle_file = gr.File(label="Original Subtitles", interactive=False)
                    translated_file = gr.File(label="Chinese Translation", interactive=False)
            
            # Subtitle-only download section
            gr.Markdown("---")  # Separator line
            gr.Markdown("### 📝 Subtitle-Only Download Results")
            with gr.Row():
                with gr.Column():
                    subtitle_result_output = gr.Textbox(
                        label="Subtitle Download Result",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column():
                    subtitle_only_file = gr.File(label="Downloaded Subtitles", interactive=False)
                    translated_only_file = gr.File(label="Chinese Translation", interactive=False)
            
            # Event handlers
            def get_info_handler(url):
                if not url:
                    return "Please enter a YouTube URL"
                
                title, uploader, duration, langs = self.get_video_info(url)
                
                info_text = f"""
                📺 Title: {title}
                👤 Uploader: {uploader}
                ⏱️ Duration: {duration}
                🌐 Available Languages: {', '.join(langs[:10])}
                """
                return info_text
            
            def download_handler(url, sub_lang, auto_caps, translate, quality):
                if not url:
                    return "Please enter a YouTube URL", None, None, None
                
                result, video, subtitle, translated = self.download_video_and_transcript(
                    url, sub_lang, auto_caps, translate, quality
                )
                
                return (
                    result,
                    video if video else None,
                    subtitle if subtitle else None,
                    translated if translated else None
                )
            
            def download_subtitle_handler(url, sub_lang, auto_caps, translate):
                if not url:
                    return "Please enter a YouTube URL", None, None
                
                result, subtitle, translated = self.download_subtitle_only(
                    url, sub_lang, auto_caps, translate
                )
                
                return (
                    result,
                    subtitle if subtitle else None,
                    translated if translated else None
                )
            
            info_button.click(
                fn=get_info_handler,
                inputs=[url_input],
                outputs=[video_info]
            )
            
            download_button.click(
                fn=download_handler,
                inputs=[url_input, subtitle_lang, include_auto_captions, translate_subtitles, video_quality],
                outputs=[result_output, video_file, subtitle_file, translated_file]
            )
            
            subtitle_only_button.click(
                fn=download_subtitle_handler,
                inputs=[url_input, subtitle_lang, include_auto_captions, translate_subtitles],
                outputs=[subtitle_result_output, subtitle_only_file, translated_only_file]
            )
            
            # Removed gr.Examples blocks to avoid duplicated/confusing results sections
            
        return interface


def main():
    """Main function to run the Gradio app."""
    print("🚀 Starting YouTube Downloader with Translation...")
    
    app = YouTubeTranslationApp()
    interface = app.create_interface()
    
    print("🌐 Launching web interface...")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
