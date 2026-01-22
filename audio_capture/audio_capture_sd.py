import numpy as np
import queue
from collections import deque
import threading
from .audio_helpers import check_power_16
import sounddevice as sd

class AudioCapture:
    def __init__(self, fft_size=4096, frames_per_buffer=1024):
        self.fft_size = fft_size
        self.frames_per_buffer = frames_per_buffer

        self.sample_rate = None
        self.channels = None

        log16 = np.log(fft_size) / np.log(16)
        if not check_power_16(fft_size):
            print(f"WARNING: FFT size {fft_size} is not a power of 16")

        self.audio_buffer = deque(maxlen=fft_size * 4)
        self.lock = threading.Lock()

        self.window = np.hanning(self.fft_size)

        self.audio_queue = queue.Queue()

        #self.p = sd # SoundDevice audioclass
        self.stream = None
        self.is_running = False
        
        sd.query_devices()
        #self.p.enumerate_devices()

    def _handle_stream(self, input_data, output_data, frames, time, status):
        """Callback for SoundDevice stream"""
        if status: print(f"Audio status: {status}")
        audio_data = np.frombuffer(input_data, dtype=np.float32)
        if self.channels == 2:
            # mono convert
            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
        self.audio_queue.put(audio_data.copy()) # add to queue
        
    def start(self, device=None):
        if self.is_running:
            raise RuntimeError("Capture already running!")

        try:
            # if no device specified find loopback
            if device is None:
                print("\nSearching for loopback devices...")
                #if (loopback_device := self.p.find_loopback_device()):
                #    device = loopback_device['index']
                #    self.sample_rate = int(loopback_device['defaultSampleRate'])
                #    self.channels = loopback_device['maxInputChannels']
                #    print(f"Found loopback device: {loopback_device['name']}")

                #    # TODO full output

                #else:
                #    print("No loopback device found!")
                default_device = sd.default.device # self.p.get_default_input_device_info()
                if default_device:
                    device = default_device[0]
                    print(f"\nFalling back to default input device {device}")

            # open audio stream
            device_info = sd.query_devices() if not isinstance(device, int) else device
            #self.p.get_device_info_by_index(device) if isinstance(device, int) else device

            self.stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.frames_per_buffer,
                    device=device,
                    channels=self.channels,
                    dtype=np.float32,
                    callback=self._handle_stream,
                    prime_output_buffers_using_stream_callback=True
            )

            self.stream.start()
            self.is_running = True

            print("Audio capture started.")

        except Exception as e:
            print(f"\nError starting audio capture: {e}")
            raise

    def stop(self):
        """Stop audio capture"""
        if self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.is_running = False
        #if self.p:
        #    self.p.abort()
        print("\nAudio capture stopped.")

