# Drew Schlesener and Denton Jarvis
# GPGPU Final Project
# Audio File Spectrogram

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html

# `module load anaconda3`
# `module keyword cuda` to see module related to keyword CUDA
# `conda env list`
# `conda create --name <env>`
# `source activate env-name` in Palmetto for conda activate
# `source activate dgl-dev-gpu-118`

import numpy as np
import cupy as cp
import soundfile as sf
import matplotlib.pyplot as plt

# 1. Load audio
data, sr = sf.read("test.wav")
if data.ndim > 1:
    data = np.mean(data, axis=1)  # make mono

# 2. Parameters
win_size = 1024
hop_size = 512
num_frames = (len(data) - win_size) // hop_size

# 3. Frame the signal on CPU
frames = np.stack([data[i*hop_size:i*hop_size+win_size] for i in range(num_frames)])
frames = np.hanning(win_size)[None, :] * frames  # apply window

# 4. Move data to GPU
frames_gpu = cp.asarray(frames, dtype=cp.complex64)

# 5. Execute FFT on GPU
spec_gpu = cp.fft.fft(frames_gpu, axis=1)

# 6. Compute magnitude on GPU
spec_magnitude = cp.abs(spec_gpu)

# 7. Copy back to CPU for plotting
spec = cp.asnumpy(spec_magnitude)

# 8. Plot
plt.figure(figsize=(10, 6))
plt.imshow(20 * np.log10(spec.T + 1e-6), origin="lower", aspect="auto", cmap="magma")
plt.xlabel("Frame")
plt.ylabel("Frequency bin")
plt.title("Spectrogram (GPU via CuPy)")
plt.colorbar(label="Magnitude (dB)")
plt.tight_layout()
plt.savefig('my_plot.jpg', dpi=150)
print(f"Spectrogram saved to my_plot.jpg")
print(f"Processed {num_frames} frames of {win_size} samples each")
print(f"Sample rate: {sr} Hz")