"""
Test script for VibeVoice TTS integration
Demonstrates basic usage and checks setup
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_vibevoice_import():
    """Test if VibeVoice can be imported"""
    print("=" * 50)
    print("Testing VibeVoice Import")
    print("=" * 50)
    
    try:
        from VITS import VibeVoiceTTS
        print("✓ VibeVoice import successful")
        return True
    except Exception as e:
        print(f"✗ VibeVoice import failed: {e}")
        return False

def test_vibevoice_initialization():
    """Test if VibeVoice can be initialized"""
    print("\n" + "=" * 50)
    print("Testing VibeVoice Initialization")
    print("=" * 50)
    
    try:
        from VITS import VibeVoiceTTS
        
        models_dir = 'checkpoints/VibeVoice'
        tts = VibeVoiceTTS(models_dir=models_dir)
        print(f"✓ VibeVoice initialized successfully")
        print(f"  Models directory: {models_dir}")
        print(f"  Device: {tts.device}")
        
        return tts
    except Exception as e:
        print(f"✗ VibeVoice initialization failed: {e}")
        return None

def test_check_models(tts):
    """Check for available models"""
    print("\n" + "=" * 50)
    print("Checking Available Models")
    print("=" * 50)
    
    try:
        models = tts.get_available_models()
        
        if models:
            print(f"✓ Found {len(models)} model(s):")
            for model in models:
                print(f"  - {model}")
        else:
            print("✗ No models found")
            print("\nTo use VibeVoice, please download models from:")
            print("  - VibeVoice-1.5B: https://huggingface.co/microsoft/VibeVoice-1.5B")
            print("  - VibeVoice-Large: https://huggingface.co/aoi-ot/VibeVoice-Large")
            print(f"\nPlace models in: {tts.models_dir}")
        
        return models
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return []

def test_check_tokenizer(tts):
    """Check for tokenizer"""
    print("\n" + "=" * 50)
    print("Checking Tokenizer")
    print("=" * 50)
    
    try:
        tokenizer_path = tts.find_tokenizer_path()
        
        if tokenizer_path:
            print(f"✓ Tokenizer found at: {tokenizer_path}")
        else:
            print("✗ Tokenizer not found")
            print("\nTo use VibeVoice, please download Qwen tokenizer from:")
            print("  https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main")
            print("\nRequired files:")
            print("  - tokenizer_config.json")
            print("  - vocab.json")
            print("  - merges.txt")
            print("  - tokenizer.json")
            print(f"\nPlace files in: {tts.models_dir / 'tokenizer'}/")
        
        return tokenizer_path is not None
    except Exception as e:
        print(f"✗ Error checking tokenizer: {e}")
        return False

def test_dependencies():
    """Check if required dependencies are installed"""
    print("\n" + "=" * 50)
    print("Checking Dependencies")
    print("=" * 50)
    
    dependencies = {
        'torch': 'torch',
        'transformers': 'transformers',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'accelerate': 'accelerate',
    }
    
    all_ok = True
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False
    
    if not all_ok:
        print("\nTo install missing dependencies:")
        print("  pip install -r VITS/requirements_vibevoice.txt")
    
    return all_ok

def print_usage_example():
    """Print usage example"""
    print("\n" + "=" * 50)
    print("Usage Example")
    print("=" * 50)
    
    example = """
# Basic usage (no voice cloning)
from VITS import VibeVoiceTTS

tts = VibeVoiceTTS(models_dir='checkpoints/VibeVoice')
tts.predict(
    text="Hello! This is a test of VibeVoice text-to-speech.",
    save_path="output.wav",
    model_name="VibeVoice-1.5B"
)

# With voice cloning
tts.predict(
    text="This will use the cloned voice.",
    save_path="output_cloned.wav",
    voice_sample_path="reference_voice.wav",
    model_name="VibeVoice-1.5B",
    seed=42,
    diffusion_steps=20,
    speed_factor=1.0
)
"""
    print(example)

def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print(" " * 15 + "VibeVoice TTS Integration Test")
    print("=" * 70)
    
    # Test import
    if not test_vibevoice_import():
        print("\n✗ Cannot proceed without VibeVoice import")
        return
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test initialization
    tts = test_vibevoice_initialization()
    if tts is None:
        print("\n✗ Cannot proceed without VibeVoice initialization")
        return
    
    # Check models
    models = test_check_models(tts)
    
    # Check tokenizer
    tokenizer_ok = test_check_tokenizer(tts)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    ready = len(models) > 0 and tokenizer_ok and deps_ok
    
    if ready:
        print("✓ VibeVoice is ready to use!")
        print(f"  Available models: {', '.join(models)}")
    else:
        print("✗ VibeVoice setup incomplete")
        
        if not deps_ok:
            print("  Missing dependencies - run: pip install -r VITS/requirements_vibevoice.txt")
        if len(models) == 0:
            print("  No models found - download from HuggingFace")
        if not tokenizer_ok:
            print("  Tokenizer not found - download Qwen tokenizer")
    
    # Print usage example
    print_usage_example()
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
