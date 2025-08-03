import os
from pathlib import Path
# Set up local nltk_data folder for tokenizers
nltk_data_path = Path(__file__).parent / 'nltk_data'
nltk_data_path.mkdir(exist_ok=True)
os.environ['NLTK_DATA'] = str(nltk_data_path)
import pysrt
import nltk
import textwrap
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
    def combine_sentence_and_length(self, input_path, output_path, max_length=60, lang='english'):
        """Combine SRT blocks by sentence, then split long sentences by max_length."""
        try:
            subs = pysrt.open(input_path)
            dictionary_path = Path(__file__).parent / 'dictionary.txt'
            preserve_phrases = set()
            if dictionary_path.exists():
                with open(dictionary_path, 'r', encoding='utf-8') as dict_file:
                    for line in dict_file:
                        line = line.strip()
                        if not line or '=' not in line:
                            continue
                        key, _ = line.split('=', 1)
                        preserve_phrases.add(key.strip())

            # Scan ahead and merge blocks until a full dictionary phrase is found
            all_text = ''
            buffer = ''
            i = 0
            while i < len(subs):
                text = subs[i].text.replace('\n', ' ')
                if buffer:
                    buffer += ' ' + text
                else:
                    buffer = text
                found_phrase = None
                for phrase in preserve_phrases:
                    if phrase in buffer:
                        found_phrase = phrase
                        break
                if found_phrase:
                    # Flush buffer as a single block
                    all_text += buffer + ' '
                    buffer = ''
                    i += 1
                    continue
                # If not found, look ahead to next block
                i += 1
            if buffer:
                all_text += buffer + ' '
            # Ensure punkt is available in local nltk_data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', download_dir=str(nltk_data_path))
            # Use 'english' for sentence splitting unless user requests Chinese and punkt/chinese is available
            lang_code = 'english'
            if lang == 'chinese':
                # Try to use Chinese sentence splitter if available, else fallback to English
                try:
                    nltk.data.find('tokenizers/punkt/chinese.pickle')
                    lang_code = 'chinese'
                except LookupError:
                    print('[combine_sentence_and_length] Warning: Chinese sentence splitter not found, using English splitter.')
            sentences = nltk.sent_tokenize(all_text, language=lang_code)
            merged = []
            idx = 0
            sub_idx = 0
            while idx < len(sentences):
                sentence = sentences[idx]
                words = re.findall(r'\w+', sentence)
                count = 0
                start, end = None, None
                text_accum = ''
                while sub_idx < len(subs) and count < len(words):
                    sub = subs[sub_idx]
                    sub_words = re.findall(r'\w+', sub.text.replace('\n', ' '))
                    if start is None:
                        start = sub.start
                    end = sub.end
                    text_accum += (sub.text + ' ')
                    count += len(sub_words)
                    sub_idx += 1
                if start is None or end is None:
                    print(f"[combine_sentence_and_length] Skipping sentence due to missing timing: {sentence}")
                    idx += 1
                    continue
                wrapped = textwrap.wrap(sentence, width=max_length)
                if len(wrapped) == 1:
                    merged.append((start, end, wrapped[0]))
                else:
                    total_ms = (end.ordinal - start.ordinal)
                    chunk_ms = total_ms // len(wrapped) if len(wrapped) > 0 else total_ms
                    for i, line in enumerate(wrapped):
                        chunk_start = start.ordinal + i * chunk_ms
                        chunk_end = chunk_start + chunk_ms
                        new_start = pysrt.SubRipTime(milliseconds=chunk_start)
                        new_end = pysrt.SubRipTime(milliseconds=chunk_end)
                        merged.append((new_start, new_end, line))
                idx += 1
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, (start, end, text) in enumerate(merged, 1):
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
            return str(output_path)
        except Exception as e:
            print(f"[combine_sentence_and_length] Error: {e}")
            return None
    def __init__(self):
        self.download_path = Path("./downloads")
        self.download_path.mkdir(exist_ok=True)
        self.translation_model = None
        self.translation_tokenizer = None
        self.setup_translation_model()
        
    def setup_translation_model(self):
        """Initialize the NLLB-200 translation model for English to Traditional Chinese, using mps on Mac if available."""
        try:
            print("Loading NLLB-200 translation model (English to Traditional Chinese)...")
            model_name = "facebook/nllb-200-3.3B"  # Full version: best quality, slower
            self.translation_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            # Device selection: mps (Mac), cuda (GPU), cpu otherwise
            import platform
            self.device_str = "cpu"
            if torch.cuda.is_available():
                self.device_str = "cuda"
            elif hasattr(torch, "has_mps") and torch.has_mps and torch.backends.mps.is_available():
                self.device_str = "mps"
            print(f"[Translation] Using device: {self.device_str}")
            if self.device_str == "cuda":
                self.translator = pipeline(
                    "translation",
                    model=self.translation_model,
                    tokenizer=self.translation_tokenizer,
                    device=0
                )
            else:
                self.translator = pipeline(
                    "translation",
                    model=self.translation_model,
                    tokenizer=self.translation_tokenizer,
                    device=-1
                )
            if self.device_str == "mps":
                self.translation_model.to("mps")
            self.src_lang = "eng_Latn"
            self.tgt_lang = "zho_Hant"
            print("✅ NLLB-200 translation model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading NLLB-200 model: {e}")
            self.translator = None
    
    def translate_text(self, text: str, max_length: int = 512) -> str:
        """Translate English text to Traditional Chinese using NLLB-200. Uses manual inference for MPS, pipeline for CUDA/CPU."""
        if not self.translator:
            return text
        try:
            sentences = re.split(r'[.!?]+', text)
            translated_sentences = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                if getattr(self, "device_str", "cpu") == "mps":
                    # Manual inference for MPS
                    inputs = self.translation_tokenizer(
                        sentence.strip(),
                        return_tensors="pt",
                        max_length=max_length,
                        padding=True
                    )
                    for k in inputs:
                        inputs[k] = inputs[k].to("mps")
                    generated_tokens = self.translation_model.generate(
                        **inputs,
                        forced_bos_token_id=self.translation_tokenizer.convert_tokens_to_ids(self.tgt_lang),
                        max_length=max_length
                    )
                    zh_text = self.translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                else:
                    # Use pipeline for CUDA/CPU
                    result = self.translator(
                        sentence.strip(),
                        max_length=max_length,
                        do_sample=False,
                        src_lang=self.src_lang,
                        tgt_lang=self.tgt_lang
                    )
                    zh_text = result[0]['translation_text']
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
        translate_subtitles: bool = True,
        max_length: int = 60
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
                            download_folder / f'{clean_title}_zh-tw.srt',
                            combine_and_split=True,
                            max_length=max_length
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
        video_quality: str = "best",
        max_length: int = 60
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
                            download_folder / f'{clean_title}_zh-tw.srt',
                            combine_and_split=True,
                            max_length=max_length
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
    
    def translate_srt_file(self, srt_path: str, output_path: Path, combine_and_split: bool = True, max_length: int = 60) -> str:
        # Load custom dictionary if present
        import re
        dictionary_path = Path(__file__).parent / 'dictionary.txt'
        custom_dict = {}
        if dictionary_path.exists():
            with open(dictionary_path, 'r', encoding='utf-8') as dict_file:
                for line in dict_file:
                    line = line.strip()
                    if not line or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    # Remove punctuation and lowercase for matching
                    key_clean = re.sub(r'[\W_]+', '', key).lower()
                    custom_dict[key_clean] = value.strip()
        """Translate an SRT subtitle file to Traditional Chinese, then combine by sentence and split by length."""
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse SRT format
            subtitle_blocks = re.split(r'\n\s*\n', content.strip())
            translated_lines = []

            for block in subtitle_blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    text_lines = lines[2:]
                    for line in text_lines:
                        if line.strip():
                            # Apply dictionary to source before translation (preserve original terms)
                            src = line.strip()
                            src_clean = re.sub(r'[\W_]+', '', src).lower()
                            for k, v in custom_dict.items():
                                if k in src_clean:
                                    # Insert original term in source
                                    src = re.sub(r'\b' + re.escape(k) + r'\b', v, src, flags=re.IGNORECASE)
                            translated_line = self.translate_text(src)
                            # Apply dictionary to translated line (for post-translation correction)
                            trans_clean = re.sub(r'[\W_]+', '', translated_line).lower()
                            for k, v in custom_dict.items():
                                if k in trans_clean:
                                    translated_line = re.sub(r'\b' + re.escape(k) + r'\b', v, translated_line, flags=re.IGNORECASE)
                            print(f"[translate_srt_file] EN: {line.strip()} -> ZHT: {translated_line}")
                            translated_lines.append(translated_line)

            # Generate temp SRT with one block per sentence and evenly distributed timing
            temp_translated_path = str(output_path) + ".tmp.srt"
            total_sentences = len(translated_lines)
            # Assume video duration is 10 minutes if unknown (600 seconds)
            default_duration = 600
            start_sec = 0
            end_sec = default_duration
            if total_sentences > 0:
                block_duration = (end_sec - start_sec) / total_sentences
            else:
                block_duration = end_sec - start_sec
            with open(temp_translated_path, 'w', encoding='utf-8') as f:
                for i, line in enumerate(translated_lines):
                    s = start_sec + i * block_duration
                    e = s + block_duration
                    # Format times as SRT
                    def sec2srt(sec):
                        h = int(sec // 3600)
                        m = int((sec % 3600) // 60)
                        s = int(sec % 60)
                        ms = int((sec - int(sec)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    f.write(f"{i+1}\n{sec2srt(s)} --> {sec2srt(e)}\n{line}\n\n")

            # Combine by sentence and split by length
            if combine_and_split:
                final_path = self.combine_sentence_and_length(temp_translated_path, output_path, max_length=max_length, lang='chinese')
                os.remove(temp_translated_path)
                return final_path
            else:
                if not translated_lines:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write("[Translation failed or no content to translate]")
                    return str(output_path)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(translated_lines))
                return str(output_path)
        except Exception as e:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"[Translation error: {e}")
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
            
            max_length_slider = gr.Slider(
                minimum=20, maximum=120, value=30, step=5,
                label="Max Characters per Subtitle Block (for Sentence Merge/Split)",
                interactive=True
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

            # Editable translation correction area and download button
            gr.Markdown("---")
            gr.Markdown("### ✏️ Post-Translation Correction (Edit before Download)")
            with gr.Row():
                with gr.Column():
                    editable_translation = gr.Textbox(
                        label="Edit Translated Subtitles (SRT)",
                        lines=20,
                        interactive=True,
                        placeholder="The translated SRT will appear here for correction after download."
                    )
                    download_corrected_btn = gr.Button("💾 Download Corrected Translation", variant="primary")
                    corrected_file = gr.File(label="Corrected Chinese Translation", interactive=False)
            
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
            
            def download_handler(url, sub_lang, auto_caps, translate, quality, max_length=60):
                if not url:
                    return "Please enter a YouTube URL", None, None, None
                result, video, subtitle, translated = self.download_video_and_transcript(
                    url, sub_lang, auto_caps, translate, quality, max_length
                )
                return (
                    result,
                    video if video else None,
                    subtitle if subtitle else None,
                    translated if translated else None
                )

            def download_subtitle_handler(url, sub_lang, auto_caps, translate, max_length=60):
                if not url:
                    return "Please enter a YouTube URL", None, None, ""
                result, subtitle, translated = self.download_subtitle_only(
                    url, sub_lang, auto_caps, translate, max_length
                )
                # Read translated SRT content for editing
                translated_content = ""
                if translated and os.path.exists(translated):
                    with open(translated, 'r', encoding='utf-8') as f:
                        translated_content = f.read()
                return (
                    result,
                    subtitle if subtitle else None,
                    translated if translated else None,
                    translated_content
                )
            
            info_button.click(
                fn=get_info_handler,
                inputs=[url_input],
                outputs=[video_info]
            )
            
            download_button.click(
                fn=download_handler,
                inputs=[url_input, subtitle_lang, include_auto_captions, translate_subtitles, video_quality, max_length_slider],
                outputs=[result_output, video_file, subtitle_file, translated_file]
            )
            
            subtitle_only_button.click(
                fn=download_subtitle_handler,
                inputs=[url_input, subtitle_lang, include_auto_captions, translate_subtitles, max_length_slider],
                outputs=[subtitle_result_output, subtitle_only_file, translated_only_file, editable_translation]
            )

            # Download corrected translation handler
            def save_corrected_translation(srt_text):
                # Save the edited SRT to a temp file for download
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(delete=False, suffix="_corrected.srt", mode="w", encoding="utf-8") as tmp:
                    tmp.write(srt_text)
                    tmp_path = tmp.name
                return tmp_path

            download_corrected_btn.click(
                fn=save_corrected_translation,
                inputs=[editable_translation],
                outputs=[corrected_file]
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
