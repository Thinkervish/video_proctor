import speech_recognition as sr
import audioop
import time
import threading

class AudioAgent:
    def __init__(self, energy_threshold=3000, noise_threshold=15000):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.noise_threshold = noise_threshold
        
        self.is_talking = False
        self.loud_noise_detected = False
        
        # Audio needs to run concurrently without blocking video frames
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.running = False

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def _listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                try:
                    # Capture brief audio chunks (1.5 seconds)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=1.5)
                    
                    # 1. Noise check using raw audio data (RMS)
                    rms = audioop.rms(audio.get_raw_data(), audio.sample_width)
                    if rms > self.noise_threshold:
                        self.loud_noise_detected = True
                        time.sleep(2)  # brief cooldown
                        self.loud_noise_detected = False
                    
                    # 2. Speech recognition check
                    # We don't care about what they said, just THAT they said something
                    text = self.recognizer.recognize_google(audio)
                    if text:
                        self.is_talking = True
                        time.sleep(1)
                        self.is_talking = False
                
                except sr.WaitTimeoutError:
                    pass  # Normal, no one was speaking
                except sr.UnknownValueError:
                    pass  # Audio was unintelligible (could just be noise)
                except Exception as e:
                    pass  # Ignore network drops from Google API

    def analyze_audio(self):
        """Returns the current state dict for the Supervisor to read"""
        return {
            "talking": self.is_talking,
            "loud_noise": self.loud_noise_detected
        }
