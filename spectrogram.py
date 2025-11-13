# Drew Schlesener and Denton Jarvis
# GPGPU Final Project
# Audio File Spectrogram

# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.specgram.html

# `module load anaconda3`
# `module keyword cuda`to see module related to keyword CUDA
# `conda env list`
# `conda create --name <env>`
# `source activate env-name` in Palmetto for conda activate
# `source activate dgl-dev-gpu-118 `

# Pip install pydub

import os
from pydub import AudioSegment

folder_path = '.'

for file_name in os.listdir(folder_path):
    if file_name.endswith('.mp3'):
        file_path = os.path.join(folder_path, file_name)
        audio = AudioSegment.from_file(file_path)
        print(f"File: {file_name}")
        print(f"Duration: {len(audio) / 1000} seconds")

