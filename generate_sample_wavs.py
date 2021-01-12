from glob import glob
import os
import random
import numpy as np
from scipy.io import wavfile
import librosa

DATA_PATH = 'KoreanEmotionSpeech'

NUM_SAMPLES = 20

raw_files = glob(os.path.join(DATA_PATH, '*/raw/*.raw'))

sample_raw_files = sorted(random.sample(raw_files, NUM_SAMPLES))

for raw_file in sample_raw_files:
    with open(raw_file, 'rb') as f:
        y = np.frombuffer(f.read(), dtype=np.int16)
        wav_file_path = os.path.join('wavs', os.path.basename(raw_file).replace('.raw', '.wav'))
        wavfile.write(wav_file_path, 16000, y)

wav_file_lists = glob('wavs/*.wav')

for f in wav_file_lists:
    y, fs = librosa.core.load(f) # sr=22050
    y = np.round(y * 2 ** 15).astype(np.int16)
    wavfile.write(f, fs, y)
 
