from glob import glob
import os
from itertools import repeat
from collections import defaultdict, Counter
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def validate_pairs(raw_file, txt_file):
    r_file = os.path.basename(raw_file)
    t_file = os.path.basename(txt_file)
    assert r_file.split('_')[0] == t_file.split('_')[0], \
           f'file name does not match {r_file} != {t_file}'

    return  True

def load_script(txt_file):
    with open(txt_file, 'r') as f:
        script = f.read().strip()
    return script

def load_meta(base_path):

    metadata = list()

    emotion_folders = glob(f'./{base_path}/*')
    for folder in emotion_folders:
        emotion = os.path.basename(folder).split('_')[0]
        raw_files = sorted(glob(os.path.join(folder, 'raw/*.raw')))
        txt_files = sorted(glob(os.path.join(folder, 'txt/*.txt')))
        for r, t in zip(raw_files, txt_files): validate_pairs(r, t)
        scripts = map(load_script, txt_files)
        meta = list(zip(raw_files, scripts, repeat(emotion)))
        metadata += meta

    return metadata

def display_meta_statistics(metadata):

    script_dict = defaultdict(list)
    for f_file, script, emo in metadata:
        script_dict[emo].append(script)
    
    for emo in script_dict:
        for e in Counter(script_dict[emo]).most_common(3): print(e) # Duplicated scripts
        num_scripts  = len(script_dict[emo])
        script_dict[emo] = set(script_dict[emo])
        num_unique_scripts = len(script_dict[emo])
        print(f'{emo}: {num_scripts} => {num_unique_scripts}')
    
    print(f"{'':^6}|{'hap':^6}|{'neu':^6}|{'ang':^6}|{'sad':^6}|{'fea':^6}|{'sur':^6}|{'dis':^6}|")
    for emo1 in script_dict:
        unions = [emo1, ]
        for emo2 in script_dict:
            unions.append(len(script_dict[emo1] & script_dict[emo2]))
        print('{:^6}|{:^6}|{:^6}|{:^6}|{:^6}|{:^6}|{:^6}|{:^6}|'.format(*unions))

    for i, key in enumerate(['neu', 'ang', 'sad', 'fea', 'sur', 'dis']):
        if i == 0: set_0 = script_dict[key]
        else: set_0 = set_0 & script_dict[key]
    print('All Union', len(set_0))

def load_raw(raw_file):
    with open(raw_file, 'rb') as f:
        y = np.frombuffer(f.read(), dtype=np.int16)
    return y

def save_wav_plot(y):
    plt.figure()
    plt.plot(y)
    plt.savefig('y.png', dpi=300)
    plt.close()
    wavfile.write('y.wav', 16000, y)
    return 

'''
For the reference encoder architecture (Figure 2), 
we use a simple 6-layer convolutional network. 
Each layer is composed of 3 × 3 filters with 2 × 2 stride, 
SAME padding and ReLU activation. 
Batch normalization (Ioffe & Szegedy, 2015) is applied to every layer. 
The number of filters in each layer doubles at half the rate of downsampling: 
32, 32, 64, 64, 128, 128.
'''

if __name__ == "__main__":
    BASE_PATH = 'KoreanEmotionSpeech'
    metadata = load_meta(BASE_PATH)

    print(random.choice(metadata))

    # display_meta_statistics(metadata)

    save_wav_plot(load_raw(random.choice(metadata)[0]))

