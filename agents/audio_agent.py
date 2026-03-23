import audioop
import threading
import time

import speech_recognition as sr


class AudioAgent:

    def __init__(
        self,
        energy_threshold: int = 3000,
        noise_threshold:  int = 15000,
    ):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.noise_threshold = noise_threshold

        # Expiry timestamps — flag is "active" while time.time() < timestamp
        self.talking_until    = 0.0
        self.loud_noise_until = 0.0

        self.running = False
        self.thread: threading.Thread | None = None

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background listening thread (idempotent)."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(
            target=self._listen_loop, daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        """Signal the listening thread to exit."""
        self.running = False

    # ─────────────────────────────────────────────────────────────
    # Background thread
    # ─────────────────────────────────────────────────────────────

    def _listen_loop(self) -> None:
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                while self.running:
                    try:
                        audio = self.recognizer.listen(
                            source,
                            timeout=1,
                            phrase_time_limit=1.5,
                        )

                        # RMS energy check — loud noise detection
                        rms = audioop.rms(
                            audio.get_raw_data(), audio.sample_width
                        )
                        if rms > self.noise_threshold:
                            self.loud_noise_until = time.time() + 2.0

                        # Speech recognition — talking detection
                        text = self.recognizer.recognize_google(audio)
                        if text:
                            self.talking_until = time.time() + 1.0

                    except sr.WaitTimeoutError:
                        pass
                    except sr.UnknownValueError:
                        pass
                    except Exception:
                        pass
        except Exception as e:
            print(f"[AudioAgent] Error initializing microphone: {e}")
            print("[AudioAgent] Please check if PyAudio is installed and a microphone is available.")
            self.running = False


    # ─────────────────────────────────────────────────────────────
    # Public API — called every frame by SupervisorAgent
    # ─────────────────────────────────────────────────────────────

    def analyze_audio(self) -> dict:
        """
        Return current audio flags.

        Returns:
            {
              "talking":    bool,   # True while within talking expiry window
              "loud_noise": bool,   # True while within noise expiry window
            }
        """
        now = time.time()
        return {
            "talking":    now < self.talking_until,
            "loud_noise": now < self.loud_noise_until,
        }