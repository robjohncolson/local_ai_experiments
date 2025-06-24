#!/usr/bin/env python3
"""
Two-Tier AI Assistant: NPU + GPU Architecture
NPU handles continuous monitoring, GPU handles complex processing when triggered.
"""

import sys
import time
import threading
import queue
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

# Audio processing
try:
    import pyaudio
    import librosa
    from scipy import signal
    print("Audio libraries imported successfully!")
except ImportError as e:
    print(f"Missing audio libraries: {e}")
    print("Install with: pip install pyaudio librosa scipy")

# AI processing
try:
    import openvino as ov
    import torch
    import whisper
    print("AI libraries imported successfully!")
except ImportError as e:
    print(f"Missing AI libraries: {e}")
    print("Install with: pip install openvino torch whisper")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AudioEvent:
    timestamp: float
    audio_data: np.ndarray
    event_type: str
    confidence: float
    metadata: Dict[str, Any] = None

@dataclass
class ProcessingStats:
    npu_activations: int = 0
    gpu_activations: int = 0
    false_positives: int = 0
    processing_time_npu: float = 0.0
    processing_time_gpu: float = 0.0
    power_saved: float = 0.0

class AudioBuffer:
    """Circular buffer for continuous audio recording"""
    
    def __init__(self, duration_seconds: int = 600, sample_rate: int = 16000):
        self.duration = duration_seconds
        self.sample_rate = sample_rate
        self.buffer_size = duration_seconds * sample_rate
        self.buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to circular buffer"""
        with self.lock:
            current_time = time.time()
            for sample in audio_chunk:
                self.buffer.append(sample)
                self.timestamps.append(current_time)
    
    def get_recent_audio(self, seconds: float) -> tuple[np.ndarray, List[float]]:
        """Get the last N seconds of audio"""
        with self.lock:
            samples_needed = int(seconds * self.sample_rate)
            recent_audio = np.array(list(self.buffer)[-samples_needed:])
            recent_timestamps = list(self.timestamps)[-samples_needed:]
            return recent_audio, recent_timestamps
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        with self.lock:
            return {
                "current_size": len(self.buffer),
                "max_size": self.buffer_size,
                "duration_seconds": len(self.buffer) / self.sample_rate,
                "oldest_timestamp": self.timestamps[0] if self.timestamps else None,
                "newest_timestamp": self.timestamps[-1] if self.timestamps else None
            }

class NPUGatekeeper:
    """Lightweight NPU-based audio monitoring"""
    
    def __init__(self):
        self.core = ov.Core()
        self.npu_device = None
        self.vad_threshold = 0.5
        self.wake_words = ["hey computer", "assistant", "help me"]
        self.stats = ProcessingStats()
        
        # Initialize NPU
        self._init_npu()
    
    def _init_npu(self):
        """Initialize NPU for audio processing"""
        devices = self.core.available_devices
        for device in devices:
            if "NPU" in device:
                self.npu_device = device
                logger.info(f"NPU found: {device}")
                break
        
        if not self.npu_device:
            logger.warning("NPU not found, using CPU for gatekeeper")
            self.npu_device = "CPU"
    
    def _create_vad_model(self):
        """Create a simple Voice Activity Detection model"""
        from openvino.runtime import opset8 as ops
        
        # Simple energy-based VAD model
        input_shape = [1, 1600]  # 0.1 seconds at 16kHz
        input_tensor = ops.parameter(input_shape, ov.Type.f32, name="audio_input")
        
        # Calculate energy
        squared = ops.multiply(input_tensor, input_tensor)
        energy = ops.reduce_mean(squared, axes=[1], keep_dims=True)
        
        # Simple threshold comparison
        threshold = ops.constant(np.array([[0.01]], dtype=np.float32), ov.Type.f32)
        is_voice = ops.greater(energy, threshold)
        
        result = ops.result(is_voice, name="voice_detected")
        model = ov.Model([result], [input_tensor], "simple_vad")
        
        return model
    
    def _extract_audio_features(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Extract features for NPU processing"""
        # Simple features: energy, zero crossing rate, spectral centroid
        features = []
        
        # Energy
        energy = np.mean(audio_chunk ** 2)
        features.append(energy)
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_chunk)[0])
        features.append(zcr)
        
        # Spectral centroid (simplified)
        if len(audio_chunk) > 0:
            fft = np.abs(np.fft.fft(audio_chunk))
            freqs = np.fft.fftfreq(len(fft))
            spectral_centroid = np.sum(freqs * fft) / np.sum(fft) if np.sum(fft) > 0 else 0
            features.append(abs(spectral_centroid))
        else:
            features.append(0)
        
        return np.array(features, dtype=np.float32)
    
    def analyze_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[AudioEvent]:
        """NPU-based analysis of audio chunk"""
        start_time = time.time()
        
        try:
            # Extract features
            features = self._extract_audio_features(audio_chunk)
            
            # Simple heuristic-based detection (would be replaced with actual NPU model)
            energy_threshold = 0.001
            zcr_threshold = 0.1
            
            is_interesting = False
            event_type = "silence"
            confidence = 0.0
            
            if features[0] > energy_threshold:  # Energy check
                if features[1] > zcr_threshold:  # Zero crossing rate suggests speech
                    is_interesting = True
                    event_type = "possible_speech"
                    confidence = min(features[0] * 100, 1.0)
                elif features[0] > energy_threshold * 5:  # High energy, might be wake word
                    is_interesting = True
                    event_type = "high_energy"
                    confidence = min(features[0] * 50, 1.0)
            
            processing_time = time.time() - start_time
            self.stats.processing_time_npu += processing_time
            
            if is_interesting:
                self.stats.npu_activations += 1
                return AudioEvent(
                    timestamp=time.time(),
                    audio_data=audio_chunk,
                    event_type=event_type,
                    confidence=confidence,
                    metadata={"features": features, "processing_time": processing_time}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"NPU analysis error: {e}")
            return None

class GPUProcessor:
    """Heavy-duty GPU processing for triggered events"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = None
        self.stats = ProcessingStats()
        
        # Initialize GPU resources
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize GPU resources"""
        try:
            # Load Whisper for speech recognition
            logger.info("Loading Whisper model...")
            self.whisper_model = whisper.load_model("base")
            logger.info(f"GPU processor initialized on {self.device}")
        except Exception as e:
            logger.error(f"GPU initialization error: {e}")
    
    def process_audio_event(self, event: AudioEvent, full_context: np.ndarray) -> Dict[str, Any]:
        """Process triggered audio event with full GPU power"""
        start_time = time.time()
        result = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "confidence": event.confidence,
            "transcription": None,
            "intent": None,
            "action": None,
            "processing_time": 0
        }
        
        try:
            # Speech-to-text with Whisper
            if self.whisper_model and event.event_type in ["possible_speech", "high_energy"]:
                logger.info("Running Whisper transcription...")
                
                # Prepare audio for Whisper
                audio_for_whisper = full_context.astype(np.float32)
                if len(audio_for_whisper) > 16000 * 10:  # Limit to 10 seconds
                    audio_for_whisper = audio_for_whisper[-16000 * 10:]
                
                # Transcribe
                whisper_result = self.whisper_model.transcribe(audio_for_whisper)
                transcription = whisper_result["text"].strip()
                
                if transcription:
                    result["transcription"] = transcription
                    result["intent"] = self._analyze_intent(transcription)
                    result["action"] = self._determine_action(result["intent"])
                    logger.info(f"Transcription: '{transcription}'")
                else:
                    # False positive
                    self.stats.false_positives += 1
                    result["action"] = "ignore"
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            self.stats.processing_time_gpu += processing_time
            self.stats.gpu_activations += 1
            
            return result
            
        except Exception as e:
            logger.error(f"GPU processing error: {e}")
            result["error"] = str(e)
            return result
    
    def _analyze_intent(self, transcription: str) -> str:
        """Simple intent analysis"""
        text = transcription.lower()
        
        if any(word in text for word in ["timer", "remind", "schedule"]):
            return "time_management"
        elif any(word in text for word in ["weather", "temperature"]):
            return "weather_query"
        elif any(word in text for word in ["search", "find", "look up"]):
            return "search_query"
        elif any(word in text for word in ["volume", "music", "play"]):
            return "media_control"
        elif any(word in text for word in ["status", "stats", "report"]):
            return "system_status"
        else:
            return "general_query"
    
    def _determine_action(self, intent: str) -> str:
        """Determine what action to take based on intent"""
        action_map = {
            "time_management": "set_timer_or_reminder",
            "weather_query": "fetch_weather",
            "search_query": "perform_search",
            "media_control": "control_media",
            "system_status": "show_status",
            "general_query": "process_general_query"
        }
        return action_map.get(intent, "unknown_intent")

class TwoTierAssistant:
    """Main coordinator for the two-tier AI system"""
    
    def __init__(self, buffer_duration: int = 600):
        self.audio_buffer = AudioBuffer(buffer_duration)
        self.npu_gatekeeper = NPUGatekeeper()
        self.gpu_processor = GPUProcessor()
        
        # Threading
        self.running = False
        self.audio_thread = None
        self.processing_thread = None
        self.event_queue = queue.Queue()
        
        # Audio setup
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000
        self.chunk_size = 1600  # 0.1 seconds
        
        # Statistics
        self.start_time = time.time()
    
    def start(self):
        """Start the two-tier assistant"""
        logger.info("Starting Two-Tier AI Assistant...")
        self.running = True
        
        # Start audio recording thread
        self.audio_thread = threading.Thread(target=self._audio_recording_loop, daemon=True)
        self.audio_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("Assistant started successfully!")
    
    def stop(self):
        """Stop the assistant"""
        logger.info("Stopping assistant...")
        self.running = False
        
        if self.audio_thread:
            self.audio_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        self.audio.terminate()
        logger.info("Assistant stopped.")
    
    def _audio_recording_loop(self):
        """Continuous audio recording with NPU monitoring"""
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            logger.info("Audio recording started")
            
            while self.running:
                # Record audio chunk
                audio_data = stream.read(self.chunk_size, exception_on_overflow=False)
                audio_chunk = np.frombuffer(audio_data, dtype=np.float32)
                
                # Add to buffer
                self.audio_buffer.add_audio(audio_chunk)
                
                # NPU analysis
                event = self.npu_gatekeeper.analyze_audio_chunk(audio_chunk)
                if event:
                    # Get extended context for GPU processing
                    context_audio, _ = self.audio_buffer.get_recent_audio(5.0)  # 5 seconds
                    event.metadata["context_audio"] = context_audio
                    
                    # Queue for GPU processing
                    self.event_queue.put(event)
                    logger.info(f"NPU detected {event.event_type} (confidence: {event.confidence:.2f})")
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
    
    def _processing_loop(self):
        """GPU processing loop for triggered events"""
        logger.info("GPU processing loop started")
        
        while self.running:
            try:
                # Wait for events from NPU
                event = self.event_queue.get(timeout=1.0)
                
                # Process with GPU
                context_audio = event.metadata.get("context_audio", event.audio_data)
                result = self.gpu_processor.process_audio_event(event, context_audio)
                
                # Execute action
                self._execute_action(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
    
    def _execute_action(self, result: Dict[str, Any]):
        """Execute the determined action"""
        action = result.get("action")
        transcription = result.get("transcription")
        
        if action == "ignore":
            logger.info("False positive detected, ignoring")
            return
        
        logger.info(f"Executing action: {action}")
        
        if action == "show_status":
            self._show_system_status()
        elif action == "set_timer_or_reminder":
            logger.info(f"Would set timer/reminder for: {transcription}")
        elif action == "fetch_weather":
            logger.info(f"Would fetch weather for: {transcription}")
        elif action == "perform_search":
            logger.info(f"Would search for: {transcription}")
        elif action == "control_media":
            logger.info(f"Would control media: {transcription}")
        else:
            logger.info(f"General response needed for: {transcription}")
    
    def _show_system_status(self):
        """Show system status and statistics"""
        uptime = time.time() - self.start_time
        buffer_info = self.audio_buffer.get_buffer_info()
        
        npu_stats = self.npu_gatekeeper.stats
        gpu_stats = self.gpu_processor.stats
        
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        print(f"Uptime: {uptime:.1f} seconds")
        print(f"Audio buffer: {buffer_info['duration_seconds']:.1f}s / {buffer_info['max_size/self.sample_rate']:.1f}s")
        print()
        print("NPU STATS:")
        print(f"  Activations: {npu_stats.npu_activations}")
        print(f"  Processing time: {npu_stats.processing_time_npu:.3f}s")
        print(f"  Avg per activation: {npu_stats.processing_time_npu/max(npu_stats.npu_activations,1)*1000:.1f}ms")
        print()
        print("GPU STATS:")
        print(f"  Activations: {gpu_stats.gpu_activations}")
        print(f"  Processing time: {gpu_stats.processing_time_gpu:.3f}s")
        print(f"  False positives: {gpu_stats.false_positives}")
        print(f"  Avg per activation: {gpu_stats.processing_time_gpu/max(gpu_stats.gpu_activations,1):.2f}s")
        print()
        print(f"Efficiency: {npu_stats.npu_activations}/{gpu_stats.gpu_activations} NPU/GPU ratio")
        print("="*50)

def main():
    """Main function"""
    print("Two-Tier AI Assistant")
    print("NPU monitors continuously, GPU processes when triggered")
    print("="*60)
    
    assistant = TwoTierAssistant()
    
    try:
        assistant.start()
        
        print("\nAssistant is running!")
        print("Try saying:")
        print("- 'Hey computer, what's the weather?'")
        print("- 'Assistant, show me the status'")
        print("- 'Help me set a timer'")
        print("\nPress Ctrl+C to stop...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        assistant.stop()
    except Exception as e:
        logger.error(f"Main error: {e}")
        assistant.stop()

if __name__ == "__main__":
    main()