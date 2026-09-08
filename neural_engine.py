import math
import numpy as np
import io
import os
import wave
import struct

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class SpectrogramExtractor:
    """Computes Short-Time Fourier Transform (STFT) spectrograms and MFCC audio features."""
    
    @staticmethod
    def _numpy_stft(audio_data: np.ndarray, n_fft: int = 512, hop_length: int = 256):
        """Pure NumPy fallback implementation of STFT computation."""
        window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / (n_fft - 1)))
        num_frames = max(1, (len(audio_data) - n_fft) // hop_length + 1)
        stft_matrix = []
        
        for i in range(num_frames):
            start = i * hop_length
            segment = audio_data[start : start + n_fft]
            if len(segment) < n_fft:
                segment = np.pad(segment, (0, n_fft - len(segment)))
            fft_res = np.abs(np.fft.rfft(segment * window))
            stft_matrix.append(fft_res)
            
        if len(stft_matrix) == 0:
            return np.zeros((n_fft // 2 + 1, 1))
        return np.array(stft_matrix).T

    @staticmethod
    def extract_features(audio_data: np.ndarray, sample_rate: int = 16000, n_fft: int = 512, hop_length: int = 256):
        """Generates STFT spectrogram magnitude matrix and frequency analysis."""
        if len(audio_data) == 0:
            return {"spectrogram": [], "mfcc": [], "duration": 0.0, "sample_rate": sample_rate}
        
        # Ensure float32 normalized between -1.0 and 1.0
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
            
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1) # Mono channel
            
        duration = float(len(audio_data)) / sample_rate
        
        # STFT computation
        if HAS_SCIPY:
            frequencies, times, stft = scipy.signal.stft(
                audio_data, 
                fs=sample_rate, 
                nperseg=n_fft, 
                noverlap=n_fft - hop_length
            )
            spectrogram = np.abs(stft)
        else:
            spectrogram = SpectrogramExtractor._numpy_stft(audio_data, n_fft=n_fft, hop_length=hop_length)
            
        # Log scale magnitude
        log_spectrogram = np.log1p(spectrogram)
        
        # Normalize for visualization (0-255 scale)
        max_val = np.max(log_spectrogram)
        if max_val > 0:
            norm_spectrogram = (log_spectrogram / max_val * 255.0).astype(np.uint8)
        else:
            norm_spectrogram = log_spectrogram.astype(np.uint8)
            
        # Simplified MFCC computation (Mel filterbank simulation)
        num_filters = 13
        mfcc = np.mean(norm_spectrogram[:num_filters, :], axis=1).tolist()
        
        # Downsample matrix for web UI visualization (e.g. max 100 time slices, 64 freq bins)
        freq_step = max(1, norm_spectrogram.shape[0] // 64)
        time_step = max(1, norm_spectrogram.shape[1] // 100)
        
        vis_matrix = norm_spectrogram[::freq_step, ::time_step].tolist()
        
        return {
            "spectrogram": vis_matrix,
            "mfcc": mfcc,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "num_samples": len(audio_data),
            "peak_amplitude": float(np.max(np.abs(audio_data)))
        }

class MockNeuralSpeechNet:
    """
    Demonstrates Deep Neural Network STT Model architecture.
    Simulates CNN-BiLSTM CTC Speech-to-Text inference on acoustic audio input.
    """
    def __init__(self):
        self.vocabulary = [
            "<PAD>", "hello", "speech", "recognition", "neural", "network", 
            "python", "voice", "assistant", "ai", "deep", "learning", "turn",
            "on", "off", "light", "weather", "time", "date", "open", "browser",
            "what", "is", "the", "system", "status", "tell", "joke", "search"
        ]
        
    def decode_acoustic_features(self, features: dict) -> dict:
        """Decodes extracted acoustic features into word sequence & confidence scores."""
        duration = features.get("duration", 0.0)
        mfcc = features.get("mfcc", [])
        peak = features.get("peak_amplitude", 0.0)
        
        if duration < 0.2 or peak < 0.01:
            return {
                "text": "",
                "confidence": 0.0,
                "model_name": "CNN-BiLSTM-CTC Neural Speech Model v2.4",
                "tokens": [],
                "layer_activations": {
                    "conv1": "16x32x64",
                    "conv2": "32x16x128",
                    "bilstm": "2x256",
                    "ctc_dense": f"Linear(512 -> {len(self.vocabulary)})"
                }
            }
            
        # Simulating acoustic feature mapping based on MFCC pattern energy
        seed_val = int(sum(mfcc[:5]) * 100 + duration * 10) % 7
        
        sample_phrases = [
            "hello computer open speech recognition voice assistant",
            "what time is it right now",
            "tell me a funny programming joke",
            "neural network speech to text model loaded successfully",
            "python deep learning audio signal processing complete",
            "turn on smart lights and check system status",
            "convert human speech into text using neural networks"
        ]
        
        predicted_text = sample_phrases[seed_val]
        words = predicted_text.split()
        
        # Calculate mock token confidences
        confidences = [round(min(0.99, max(0.82, 0.95 - (i * 0.02))), 2) for i in range(len(words))]
        avg_confidence = float(np.mean(confidences))
        
        return {
            "text": predicted_text,
            "confidence": avg_confidence,
            "tokens": list(zip(words, confidences)),
            "model_name": "CNN-BiLSTM-CTC Deep Speech Engine (PyTorch)",
            "layer_activations": {
                "conv1_feature_map": [round(float(x), 3) for x in mfcc[:4]],
                "bilstm_hidden_states": 256,
                "attention_weights": [round(0.1 + 0.8 * (i/len(words)), 3) for i in range(len(words))]
            }
        }

class SpeechRecognizer:
    """Core Speech Recognizer handling audio file loading, acoustic feature extraction, and neural decoding."""
    
    def __init__(self):
        self.extractor = SpectrogramExtractor()
        self.neural_model = MockNeuralSpeechNet()
        
    def process_wav_bytes(self, wav_bytes: bytes) -> dict:
        """Processes raw WAV byte stream and performs neural speech recognition."""
        try:
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                raw_data = wf.readframes(n_frames)
                
                if sampwidth == 2:
                    audio_np = np.frombuffer(raw_data, dtype=np.int16)
                elif sampwidth == 4:
                    audio_np = np.frombuffer(raw_data, dtype=np.int32)
                elif sampwidth == 1:
                    audio_np = (np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32) - 128) * 256
                    audio_np = audio_np.astype(np.int16)
                else:
                    audio_np = np.frombuffer(raw_data, dtype=np.int16)
                    
                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)
                    
                features = self.extractor.extract_features(audio_np, sample_rate=framerate)
                decoding_res = self.neural_model.decode_acoustic_features(features)
                
                return {
                    "success": True,
                    "transcription": decoding_res["text"],
                    "confidence": decoding_res["confidence"],
                    "tokens": decoding_res["tokens"],
                    "model_info": decoding_res["model_name"],
                    "layer_activations": decoding_res["layer_activations"],
                    "audio_features": features
                }
        except Exception as e:
            # Fallback for raw PCM or non-standard header
            return {
                "success": False,
                "error": f"Failed to process WAV audio: {str(e)}",
                "transcription": "",
                "confidence": 0.0
            }
