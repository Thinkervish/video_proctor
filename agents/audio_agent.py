import speech_recognition as sr
import audioop
import time
import threading

class AudioAgent:
    def __init__(self, energy_threshold=3000, noise_threshold=15000):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.noise_threshold = noise_threshold

        self.talking_until = 0        # ← timestamp, not boolean
        self.loud_noise_until = 0     # ← timestamp, not boolean

        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                try:
                    audio = self.recognizer.listen(
                        source, timeout=1, phrase_time_limit=1.5
                    )

                    rms = audioop.rms(
                        audio.get_raw_data(), audio.sample_width
                    )
                    if rms > self.noise_threshold:
                        # ← set expiry timestamp, no sleep needed
                        self.loud_noise_until = time.time() + 2

                    text = self.recognizer.recognize_google(audio)
                    if text:
                        # ← set expiry timestamp, no sleep needed
                        self.talking_until = time.time() + 1

                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except Exception:
                    pass

    def analyze_audio(self):
        now = time.time()
        return {
            # ← True only while within the expiry window
            "talking":     now < self.talking_until,
            "loud_noise":  now < self.loud_noise_until,
        }