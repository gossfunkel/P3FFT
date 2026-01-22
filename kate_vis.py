import numpy as np
import sounddevice as sd
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
        Shader, ShaderBuffer, GeomEnums, CardMaker
)

from fft import Radix16FFT, CastBuffer

# 1) generate frequency information
# 2) iFFT
# 3) send audio data to sounddevice

# a generic vertex shader for a card
CARD_VTX = """
#version 430

uniform mat4 p3d_ModelViewProjectionMatrix;

in vec4 p3d_Vertex;
in vec2 texcoord;

out vec2 vtexcoord;

void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    vtexcoord = texcoord;
}
""".strip()

# a circular visualiser for waves, using data from the SSBO
CARD_FRG = """
#version 430
layout (std430, binding = 0) buffer ssbo { float signal[]; };

in vec2 vtexcoord;

out vec4 p3d_FragColor;

const float TAU = 6.283185307179586;

vec2 cart_to_polar( vec2 cartesian ) {
    // distance from origin via pythagoras
    float x = cartesian.x - .5;
    float y = cartesian.y - .5;
    float rad = sqrt(x*x + y*y);
    // angle via euler
    float theta = atan(y/x);
    // return angle normalised to 1 radian per rotation
    return vec2(rad,theta/2);
}

void main() {
    vec2 uv = fract(cart_to_polar(vtexcoord));
    //vec2 uv = vtexcoord;
    uint idx = uint(uv.y*signal.length());
    float val = max(0., signal[idx] - uv.x + (.5 - uv.x));
    p3d_FragColor = vec4(val-uv.x*4, -uv.x, val/4. - uv.x*8., 1.);
}
""".strip()

CARD_SHDR = Shader.make(Shader.SL_GLSL, vertex=CARD_VTX, fragment= CARD_FRG)


class FFTSynth:
    def __init__(self, sample_rate = 44100, frames_per_buff = 1024, fft_size=4096):
        self.sample_rate = sample_rate
        self.frames_per_buff = frames_per_buff
        self.fft_size = fft_size

        # get the default audio device from sounddevice
        self.device = sd.default.device

        self.freq = 100 # test tone
        # generate an empty buffer
        self.signal = np.zeros(self.fft_size, dtype=np.float32)
        # set tones
        for i in range(6):
            self.signal[self.freq*(i+1)] = np.float32(1.)

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

        # set up the visualiser card
        cm = CardMaker("screen_card")
        cm.setFrameFullscreenQuad()
        self.card = base.render2d.attach_new_node(cm.generate())
        self.card.set_shader(CARD_SHDR)
        # make sure to pass it an ssbo- the fft gives us one!
        self.card.set_shader_input("ssbo", self.gpu_handle.buffer)

        # start tasks and audio stream
        base.taskMgr.add(self._load_buff, "load_buffer", sort=10)
        self.stream.start()
        base.taskMgr.add(self._load_fft, "load_fft", sort=20)

    def _load_buff(self, task):
        # get the data from the SSBO for the audio output stream
        self.audio_buff[:] = self.fft.fetch(self.gpu_handle)
        return task.cont
    
    def _load_fft(self, task):
        # add more tones for some variety ;P
        self.freq += int(np.sin(task.frame)*40)
        self.signal[self.freq] += 1
        # run an inverse dft on the sample (frequency data)
        self.gpu_handle = self.fft.fft(self.signal, True)
        return task.cont

    def _init_load_fft(self):
        # prime the fft with an initial run and return the handle
        sig_buffer = ShaderBuffer("signal", self.signal.tobytes(), GeomEnums.UH_stream)
        gpu_handle = CastBuffer(sig_buffer, self.fft_size, cast=np.float32)
        return self.fft.fft(gpu_handle, True)

    def __del__(self):
        self.stream.stop()
        self.stream.close()

if __name__ == "__main__":
    # panda3d init
    ShowBase()
    # init my synth thing
    FFTSynth()
    # run panda3d (and the tasks and nodes ive loaded into it)
    base.run()

    # make sure that stream is closed!
    FFTSynth.stream.abort()


