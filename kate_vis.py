import numpy as np
import sounddevice as sd
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
        Shader, ShaderBuffer, GeomEnums, CardMaker
)
from collections import deque
from fft import Radix16FFT, CastBuffer

TAU: float = 2 * np.pi

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
    float theta = atan(y/x); //cos(x) + sin(y);
    // return angle normalised to 1 radian per rotation
    return vec2(rad,theta/2);
}

vec2 polar_to_cart( vec2 polar ) {
    // origin is in the centre
    float x = polar.x * cos(polar.y);
    float y = polar.x * sin(polar.y);
    return vec2(x,y);
}

vec3 render_circular( vec2 texc ) {
    vec2 uvcirc = cart_to_polar(texc);
    uint idx = uint(uvcirc.y*signal.length());
    float amp = signal[idx];
    float val = amp - uvcirc.x;
    return vec3(val-uvcirc.x*1.5, val-uvcirc.x, val - uvcirc.x*1.45);
}

vec3 render_barchart( vec2 uv ) {
    uint idx = uint(uv.x*1024);
    float val = max(uv.y, signal[idx]) -1;
    return vec3(val-uv.y*1.5, val-uv.y, val - uv.y*1.45);
}

void main() {
    vec3 col = render_barchart(vtexcoord);
    p3d_FragColor = vec4(col, 1.);
}
""".strip()

CARD_SHDR = Shader.make(Shader.SL_GLSL, vertex=CARD_VTX, fragment= CARD_FRG)


class FFTSynth:
    def __init__(self, sample_rate = 48000, frames_per_buff = 1024, fft_size=4096):
        self.sample_rate: int = sample_rate
        self.frames_per_buff: int = frames_per_buff
        self.fft_size: int = fft_size

        # get the default audio device from sounddevice
        self.device: int = sd.default.device

        self.freq: int = 320 # test tone
        self.filter_freq: int = 120 # FFT filter
        self.amp: float = .7 # chill out a bit ! protect speakers
        # generate a tone buffer for the fft
        self.signal = self._init_gen_tone()

        # initialise an empty audio buffer for data from the fft 
        self.audio_buff_max_len: int = frames_per_buff * 12
        self.audio_buff = np.zeros(self.fft_size, dtype=np.float32)

        def _callb(output_data, frames, time, status): 
            # load audio from buffer from the fft into stream
            output_data[:, 0] = self.audio_buff[:frames_per_buff]
            # pop the 'consumed' frames
            self.audio_buff = self.audio_buff[frames_per_buff:]

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
        # load some data into the audio buffer and ready the next tone data for the fft
        self._init_load_buff()
        self._init_gen_tone(self.fft_size)
        self.gpu_handle = self._init_load_fft()
        self._init_load_buff()
        self._init_gen_tone(self.fft_size*2)

        # set up the visualiser card
        cm = CardMaker("screen_card")
        cm.setFrameFullscreenQuad()
        self.card = base.render2d.attach_new_node(cm.generate())
        self.card.set_shader(CARD_SHDR)
        # make sure to pass it an ssbo- the fft gives us one!
        self.card.set_shader_input("ssbo", self.gpu_handle.buffer)

        # start tasks and audio stream
        base.taskMgr.add(self._load_buff, "load_buffer", sort=5)
        self.stream.start()

        # start the tone generator and the FFTs
        base.taskMgr.add(self._gen_tone, "gen_tone", sort=15)
        base.taskMgr.add(self._load_fft, "load_fft", sort=25)

    def _gen_tone(self, task):
        phase: float = self.fft_size * (task.frame + 3)
        freq = self.freq * (1.5 + np.sin(40. * task.frame))
        t = np.linspace(0,1,self.fft_size, dtype=np.float64)
        self.sample = np.array(np.sin(TAU * t * freq + phase), 
                                dtype=np.complex64)
        self.sample *= self.amp
        return task.cont

    def _init_gen_tone(self, phase: float = 0):
        t = np.linspace(0,1,self.fft_size, dtype=np.float64)
        sample = np.sin(TAU * t * self.freq + phase)
        sample *= self.amp
        return np.array(sample, dtype=np.complex64)

    def _load_buff(self, task):
        # get the data from the SSBO for the audio output stream
        from_fft = np.array(self.fft.fetch(self.gpu_handle), dtype=np.float32)
        # add to back end of buffer
        self.audio_buff = np.append(self.audio_buff, from_fft)

        # only add to buffer if shorter than max size
        #sum_buff_lengths = len(self.audio_buff) + len(from_fft)
        #if sum_buff_lengths > self.audio_buff_max_len:
        #    diff = self.audio_buff_max_len - len(self.audio_buff)
        #    from_fft = from_fft[:diff]

        # update the card
        self.card.set_shader_input("ssbo", self.gpu_handle.buffer)
        return task.cont

    def _init_load_buff(self):
        # get the data from the SSBO for the audio output stream
        from_fft = np.array(self.fft.fetch(self.gpu_handle), dtype=np.float32)
        self.audio_buff = np.append(self.audio_buff, from_fft)
    
    def _load_fft(self, task):
        self.gpu_handle = self.fft.fft(self.signal, False)
        freqdata = self.fft.fetch(self.gpu_handle)
        freqdata = freqdata / 2
        freqdata[self.filter_freq:] = np.zeros(len(freqdata) - self.filter_freq)
        # run an inverse dft on the sample (frequency data)
        self.gpu_handle = self.fft.fft(freqdata, True)
        return task.cont

    def _init_load_fft(self):
        # prime the fft with an initial run and return the handle
        sig_buffer = ShaderBuffer("signal", self.signal.tobytes(), GeomEnums.UH_stream)
        gpu_handle = CastBuffer(sig_buffer, self.fft_size, cast=np.complex64)
        gpu_handle = self.fft.fft(gpu_handle, False)
        freqdata = self.fft.fetch(gpu_handle)
        freqdata = freqdata/2
        freqdata[self.filter_freq:] = np.zeros(len(freqdata) - self.filter_freq)
        return self.fft.fft(freqdata, True)

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


