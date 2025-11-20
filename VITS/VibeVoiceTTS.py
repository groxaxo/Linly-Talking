"""
VibeVoice TTS Integration for Linly-Talker
Realtime voice cloning using Microsoft's VibeVoice model
Based on: https://github.com/groxaxo/VibeVoice-FastAPI
"""

import os
import sys
import logging
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VibeVoiceTTS")


def get_optimal_device():
    """Get the best available device (cuda, mps, or cpu)"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class VibeVoiceTTS:
    """
    VibeVoice TTS with realtime voice cloning support
    Uses Microsoft's VibeVoice models for natural speech synthesis
    """
    
    def __init__(self, models_dir: str = "./checkpoints/VibeVoice"):
        """
        Initialize VibeVoice TTS
        
        Args:
            models_dir: Directory containing VibeVoice models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.processor = None
        self.current_model = None
        self.device = get_optimal_device()
        
        logger.info(f"VibeVoice TTS initialized on device: {self.device}")
        logger.info(f"Models directory: {self.models_dir}")
    
    def get_available_models(self):
        """Get list of available VibeVoice models"""
        models = []
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory does not exist: {self.models_dir}")
            return models
        
        for item in self.models_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if (item / "config.json").exists():
                    models.append(item.name)
        
        models.sort()
        return models
    
    def find_tokenizer_path(self) -> Optional[Path]:
        """Find Qwen tokenizer path"""
        # Check tokenizer folder
        tokenizer_dir = self.models_dir / "tokenizer"
        if tokenizer_dir.exists():
            required_files = ["tokenizer_config.json", "vocab.json", "merges.txt"]
            if all((tokenizer_dir / f).exists() for f in required_files):
                return tokenizer_dir
        
        # Check HuggingFace cache
        hf_cache_paths = [
            Path.home() / ".cache/huggingface/hub",
        ]
        
        if "HF_HOME" in os.environ:
            hf_cache_paths.insert(0, Path(os.environ["HF_HOME"]) / "hub")
        
        for cache_path in hf_cache_paths:
            if cache_path.exists():
                qwen_cache = cache_path / "models--Qwen--Qwen2.5-1.5B"
                if qwen_cache.exists():
                    snapshots_dir = qwen_cache / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir() and (snapshot / "tokenizer_config.json").exists():
                                return snapshot
        
        return None
    
    def load_model(self, model_name: str = "VibeVoice-1.5B"):
        """
        Load VibeVoice model
        
        Args:
            model_name: Name of the model to load (VibeVoice-1.5B or VibeVoice-Large)
        """
        if self.model is not None and self.current_model == model_name:
            logger.info(f"Model {model_name} already loaded")
            return
        
        model_path = self.models_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Available models: {self.get_available_models()}\n"
                f"Please download the model from HuggingFace:\n"
                f"  - VibeVoice-1.5B: https://huggingface.co/microsoft/VibeVoice-1.5B\n"
                f"  - VibeVoice-Large: https://huggingface.co/aoi-ot/VibeVoice-Large"
            )
        
        try:
            # Import VibeVoice
            from transformers import AutoModel, AutoProcessor
            import warnings
            warnings.filterwarnings("ignore")
            
            logger.info(f"Loading VibeVoice model: {model_name}")
            
            # Load model
            self.model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                local_files_only=True,
            )
            
            # Load processor
            tokenizer_path = self.find_tokenizer_path()
            if not tokenizer_path:
                raise FileNotFoundError(
                    "Qwen tokenizer not found!\n"
                    "Please download from: https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main\n"
                    "Required files: tokenizer_config.json, vocab.json, merges.txt, tokenizer.json\n"
                    f"Place in: {self.models_dir / 'tokenizer'}/"
                )
            
            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                local_files_only=True,
                language_model_pretrained_name=str(tokenizer_path),
            )
            
            self.current_model = model_name
            logger.info(f"Model {model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_voice_sample(self, audio_path: str, speed_factor: float = 1.0) -> np.ndarray:
        """
        Prepare voice sample from audio file
        
        Args:
            audio_path: Path to audio file
            speed_factor: Speed adjustment factor (0.8-1.2)
        
        Returns:
            Voice sample as numpy array at 24kHz
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Resample to 24kHz
            if sr != 24000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            
            # Apply speed change if needed
            if speed_factor != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            
            # Ensure mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)
            
            # Normalize
            if audio.max() > 0:
                audio = audio / np.abs(audio).max() * 0.95
            
            return audio
            
        except Exception as e:
            logger.error(f"Failed to prepare voice sample: {e}")
            return None
    
    def predict(self, text: str, save_path: str = "vibevoice_output.wav",
                voice_sample_path: Optional[str] = None,
                model_name: str = "VibeVoice-1.5B",
                seed: int = 42,
                diffusion_steps: int = 20,
                cfg_scale: float = 1.3,
                speed_factor: float = 1.0):
        """
        Generate speech from text with optional voice cloning
        
        Args:
            text: Input text to synthesize
            save_path: Output audio file path
            voice_sample_path: Optional path to reference voice audio for cloning
            model_name: Model to use (VibeVoice-1.5B or VibeVoice-Large)
            seed: Random seed for reproducibility
            diffusion_steps: Number of diffusion steps (higher = better quality, slower)
            cfg_scale: Classifier-free guidance scale
            speed_factor: Voice speed adjustment (0.8-1.2)
        
        Returns:
            Path to generated audio file
        """
        # Load model if not loaded
        if self.model is None or self.current_model != model_name:
            self.load_model(model_name)
        
        try:
            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # Prepare voice sample if provided
            voice_sample = None
            if voice_sample_path and os.path.exists(voice_sample_path):
                voice_sample = self.prepare_voice_sample(voice_sample_path, speed_factor)
                if voice_sample is not None:
                    logger.info(f"Using voice cloning from: {voice_sample_path}")
            
            # Format text for single speaker
            formatted_text = f"Speaker 1: {text}"
            
            # Prepare inputs
            voice_samples = [voice_sample] if voice_sample is not None else None
            
            inputs = self.processor(
                [formatted_text],
                voice_samples=voice_samples if voice_samples else None,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move inputs to device
            if self.device == "cuda":
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Set diffusion steps
            self.model.set_ddpm_inference_steps(diffusion_steps)
            
            # Generate
            logger.info(f"Generating speech with {diffusion_steps} steps...")
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    guidance_scale=cfg_scale,
                    max_new_tokens=2048,
                    use_sampling=False,
                )
            
            # Extract audio
            audio_tensor = output.audio_values
            
            # Convert to numpy
            if audio_tensor.dim() == 3:
                audio = audio_tensor.squeeze().cpu().numpy()
            else:
                audio = audio_tensor.cpu().numpy()
            
            # Save audio
            sf.write(save_path, audio, 24000)
            logger.info(f"Audio saved to: {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise
    
    def free_memory(self):
        """Free model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model memory freed")


if __name__ == "__main__":
    # Test VibeVoice TTS
    tts = VibeVoiceTTS()
    
    # Check available models
    models = tts.get_available_models()
    print(f"Available models: {models}")
    
    # Generate speech
    if models:
        tts.predict(
            text="Hello! This is a test of VibeVoice text-to-speech with realtime voice cloning capabilities.",
            save_path="test_output.wav",
            model_name=models[0]
        )
        print("Test completed successfully!")
    else:
        print("No models found. Please download VibeVoice models first.")
