import numpy as np
import sounddevice as sd
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
        ShaderBuffer, GeomEnums
)

from fft import Radix16FFT, CastBuffer

# 1) generate frequency information
# 2) iFFT
# 3) send audio data to sounddevice


class FFTSynth:
    def __init__(self, sample_rate = 44100, frames_per_buff = 1024, fft_size=4096):
        self.sample_rate = sample_rate
        self.frames_per_buff = frames_per_buff
        self.fft_size = fft_size

        # get the default audio device from sounddevice
        self.device = sd.default.device

        self.freq = 440 # test tone
        # generate an empty buffer
        self.signal = np.zeros(self.fft_size, dtype=np.float32)
        # set one frequency high
        self.signal[self.freq] = np.float32(1.)

        # initialise an empty audio buffer for data from the fft
        self.audio_buff = np.zeros(self.fft_size, dtype=np.float32)

        def _callb(output_data, frames, time, status): 
            # load audio from buffer from the fft into stream
            output_data[:, 0] = self.audio_buff[:frames_per_buff]
            # loop the 'consumed' frames from start to the end of the buffer
            self.audio_buff = np.append(self.audio_buff[frames_per_buff:],
                                        self.audio_buff[:frames_per_buff])

        # initialise the audio output stream
        self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                blocksize =self.frames_per_buff,
                device    =self.device,
                dtype     =np.float32,
                callback  =_callb,
                prime_output_buffers_using_stream_callback=True
            )

        # set up the fft and SSBO handle
        self.fft = Radix16FFT(base)
        self.gpu_handle = self._init_load_fft()

        # start tasks and audio stream
        base.taskMgr.add(self._load_buff, "load_buffer", sort=10)
        self.stream.start()
        base.taskMgr.add(self._load_fft, "load_fft", sort=20)

    def _load_buff(self, task):
        # get the data from the SSBO for the audio output stream
        self.audio_buff[:] = self.fft.fetch(self.gpu_handle)
        return task.cont
    
    def _load_fft(self, task):
        # run an inverse dft on the sample (frequency data)
        self.gpu_handle = self.fft.fft(self.signal, True)
        return task.cont

    def _init_load_fft(self):
        # prime the fft with an initial run and return the handle
        sig_buffer = ShaderBuffer("signal", self.signal.tobytes(), GeomEnums.UH_stream)
        gpu_handle = CastBuffer(sig_buffer, self.fft_size, cast=np.float32)
        return self.fft.fft(gpu_handle, True)

if __name__ == "__main__":
    ShowBase()
    FFTSynth()
    base.run()

