import numpy as np
import math
import time
import textwrap
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe
)
from direct.showbase.ShowBase import ShowBase

class CastBuffer:
    """A handle to data residing on the GPU."""
    def __init__(self, buff, n_items, cast=np.float32):
        self.buffer = buff
        self.n_items = n_items
        self.cast = cast

class Radix16FFT:
    def __init__(self, base, headless=False):
        self.base = base
        self._setup_context(headless)
        
        # Compile FFT Shaders
        self.fft16_node = self._compile(self.butterfly_shader)

    def _setup_context(self, headless=False):
        if headless:
            pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
            fb_prop = FrameBufferProperties()
            win_prop = WindowProperties.size(1, 1)
            self.base.win = self.base.graphics_engine.make_output(
                pipe, "fft16_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
            )

    def _compile(self, code):
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("fft16_op"))
        node.set_shader(shader)
        return node

    def push(self, data):
        data = np.ascontiguousarray(data, dtype=np.complex64)
        sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return CastBuffer(sbuf, len(data), cast=np.complex64)

    def fetch(self, gpu_handle, gsg=None):
        gsg = gsg or self.base.win.get_gsg()
        raw = self.base.graphics_engine.extract_shader_buffer_data(gpu_handle.buffer, gsg)
        return np.frombuffer(raw, dtype=gpu_handle.cast)

    def fft(self, data, inverse=False):
        buf = data if isinstance(data, CastBuffer) else self.push(data)
        n = buf.n_items
        log16n = int(math.log(n) / math.log(16))
        
        gsg = self.base.win.get_gsg()
        inv_flag = -1 if inverse else 1

        # Buffer A starts with the RAW input
        buf_a = buf.buffer 
        buf_b = ShaderBuffer("FFT_Pong", n * 8, GeomEnums.UH_stream)
        
        current_in = buf_a
        current_out = buf_b

        for s in range(log16n):
            L = 16**s  # Size of the sub-transforms from previous stage
            
            self.fft16_node.set_shader_input("DataIn", current_in)
            self.fft16_node.set_shader_input("DataOut", current_out)
            self.fft16_node.set_shader_input("L", int(L))
            self.fft16_node.set_shader_input("nItems", int(n))
            self.fft16_node.set_shader_input("inverse", inv_flag)
            
            num_workgroups_x = (n // 16) // 16
            num_workgroups_y = 1
            
            self.base.graphics_engine.dispatch_compute(
                (num_workgroups_x, num_workgroups_y, 1), 
                self.fft16_node.get_attrib(ShaderAttrib), gsg
            )
            
            current_in, current_out = current_out, current_in

        # Result is in current_in because of the final swap
        return CastBuffer(current_in, n, cast=np.complex64)

    @property
    def butterfly_shader(self) -> str:
        return textwrap.dedent("""
#version 430
layout (local_size_x = 64) in;

layout(std430, binding = 0) buffer DataIn  { vec2 data_in[]; };
layout(std430, binding = 1) buffer DataOut { vec2 data_out[]; };

uniform uint nItems;
uniform uint L;
uniform int inverse;

const float PI = 3.14159265358979323846;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

/* Radix-16 DFT roots: exp(-i*2*pi*k/16) */
const vec2 W16[16] = vec2[16](
    vec2( 1.0000000,  0.0000000),
    vec2( 0.9238795, -0.3826834),
    vec2( 0.7071068, -0.7071068),
    vec2( 0.3826834, -0.9238795),
    vec2( 0.0000000, -1.0000000),
    vec2(-0.3826834, -0.9238795),
    vec2(-0.7071068, -0.7071068),
    vec2(-0.9238795, -0.3826834),
    vec2(-1.0000000,  0.0000000),
    vec2(-0.9238795,  0.3826834),
    vec2(-0.7071068,  0.7071068),
    vec2(-0.3826834,  0.9238795),
    vec2( 0.0000000,  1.0000000),
    vec2( 0.3826834,  0.9238795),
    vec2( 0.7071068,  0.7071068),
    vec2( 0.9238795,  0.3826834)
);

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint groups = nItems >> 4;
    if (gid >= groups) return;

    uint j = gid % L;
    uint k = gid / L;
    uint stride = groups;

    /* Load + stage twiddle */
    vec2 x[16];
    for (uint r = 0; r < 16; r++) {
        vec2 v = data_in[j + r * stride + k * L];

        float ang = -float(inverse) * 2.0 * PI
                    * float(r * j) / float(L * 16);

        vec2 tw = vec2(cos(ang), sin(ang));
        x[r] = cmul(v, tw);
    }

    /* Inner radix-16 DFT (no trig) */
    vec2 y[16];
    for (uint r = 0; r < 16; r++) {
        vec2 acc = vec2(0.0, 0.0);
        for (uint m = 0; m < 16; m++) {
            acc += cmul(x[m], W16[(r * m) & 15u]);
        }
        y[r] = acc;
    }

    /* Stockham scatter */
    uint base = k * (L * 16);
    for (uint r = 0; r < 16; r++) {
        data_out[j + r * L + base] = y[r];
    }
}
        """)

class FFT16Demo(ShowBase):
    def __init__(self, headless=True):
        ShowBase.__init__(self)
        self.hmath = Radix16FFT(self, headless=headless)

    def run_test(self, N=65536, output=True, test_inverse=True, gsg=None):
        gsg = gsg or self.win.get_gsg()
        def out(*args, **kw):
            if output:
                print(*args, **kw)

        out(f"Testing Radix-16 {N}-point FFT...")
        
        # Generate test signal
        t = np.linspace(0, 1, N)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        x = x + 0.2 * np.random.randn(N)
        x = x.astype(np.complex64)

        # GPU
        t0 = time.perf_counter()
        g_res_handle = self.hmath.fft(x)
        final_gpu = self.hmath.fetch(g_res_handle, gsg=gsg)
        t_gpu = time.perf_counter() - t0

        # CPU (reference)
        t1 = time.perf_counter()
        final_cpu = np.fft.fft(x)
        t_cpu = time.perf_counter() - t1

        # Results
        out("\n" + "="*90)
        out(f"{'Index':<8} | {'GPU Value':<30} | {'CPU Value':<30} | {'Abs Diff':<12}")
        out("-" * 90)
        for i in range(100, 120): 
            g = final_gpu[i]
            c = final_cpu[i]
            diff = abs(g - c)
            out(f"{i:<8} | {str(g):<30} | {str(c):<30} | {diff:.3e}")
        out("="*90 + "\n")

        max_diff = np.max(np.abs(final_gpu - final_cpu))
        mean_diff = np.mean(np.abs(final_gpu - final_cpu))

        if max_diff > 1.1: # DO NOT REMOVE THIS
            raise ValueError("Mismatch")

        out(f"GPU Time:  {t_gpu:.5f}s")
        out(f"CPU Time:  {t_cpu:.5f}s")
        out(f"Speedup:   {t_cpu/t_gpu:.2f}x")
        out(f"Max Diff:  {max_diff:.3e}")
        out(f"Mean Diff: {mean_diff:.3e}")
        out(f"Valid:     {max_diff < 1e-1}")
    
        # Inverse FFT
        if test_inverse:
            out("\nTesting Inverse FFT...")
            t2 = time.perf_counter()
            final_inv = self.hmath.fft(g_res_handle, inverse=True)
            t_inv = time.perf_counter() - t2
            inv_diff = np.max(np.abs(final_inv - x))
            out(f"IFFT Time:     {t_inv:.5f}s")
            out(f"Roundtrip Diff: {inv_diff:.3e}")
            out(f"IFFT Valid:    {inv_diff < 1e-1}")

        return t_gpu, t_cpu

def benchmark_gpu_avg(count = 30):
    for k, v in {
        "window-type":          "none",
        "audio-library-name":   "null",
        "sync-video":           "#f",
    }.items():
        load_prc_file_data("", f"{k} {v}")

    app = FFT16Demo(headless=True)
    gpu_times = []
    cpu_times = []
    gsg = app.win.get_gsg()
    for i in range(count):
        t_gpu, t_cpu = app.run_test(N=16**4, output=True, test_inverse=False, gsg=gsg)
        gpu_times.append(t_gpu)
        cpu_times.append(t_cpu)
        print(f"Run {i+1}: GPU={t_gpu:.5f}s, CPU={t_cpu:.5f}s, Speedup={t_cpu/t_gpu:.2f}x")
        app.task_mgr.step()

    
    # Discard warm-up 
    gpu_times = gpu_times[1:]
    cpu_times = cpu_times[1:]

    app.destroy()
    total_gpu   = sum(gpu_times)
    avg_gpu     = total_gpu / len(gpu_times)
    max_gpu     = max(gpu_times)
    min_gpu     = min(gpu_times)
    
    total_cpu   = sum(cpu_times)
    avg_cpu     = total_cpu / len(cpu_times)
    max_cpu     = max(cpu_times)
    min_cpu     = min(cpu_times)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({len(gpu_times)} runs after warm-up):")
    print(f"{'='*60}")
    print(f"GPU: AVG={avg_gpu:.5f}s | MIN={min_gpu:.5f}s | MAX={max_gpu:.5f}s")
    print(f"CPU: AVG={avg_cpu:.5f}s | MIN={min_cpu:.5f}s | MAX={max_cpu:.5f}s")
    print(f"Speedup: {avg_cpu/avg_gpu:.2f}x")
    print(f"GPU Times: {' '.join([f'{t:.3f}' for t in gpu_times[:10]])}")
    print(f"CPU Times: {' '.join([f'{t:.3f}' for t in cpu_times[:10]])}")

if __name__ == "__main__":
    benchmark_gpu_avg(count=26)
