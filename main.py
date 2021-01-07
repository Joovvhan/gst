from glob import glob
import os
from itertools import repeat
from collections import defaultdict, Counter
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

import json
import csv

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch.utils.data import DataLoader

from torch.nn.utils.rnn import pad_sequence

from mel2samp import MEL2SAMP

from tensorboardX import SummaryWriter

# torch.cuda.amp
from torch.cuda.amp import autocast

EMO_LIST = ['neu', 'hap', 'ang', 'sad', 'fea', 'sur', 'dis']
EMO_DICT = {emo: i for i, emo in enumerate(EMO_LIST)}

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

    '''
    UserWarning: The given NumPy array is not writeable, 
    and PyTorch does not support non-writeable tensors. 
    This means you can write to the underlying (supposedly non-writeable) 
    NumPy array using the tensor. You may want to copy the array 
    to protect its data or make it writeable before converting it to a tensor. 
    This type of warning will be suppressed for the rest of this program.
    '''

    return np.array(y, copy=True)

def save_wav_plot(mel, y):
    mel = mel_normalize(mel)
    fig, axes = plt.subplots(2, 1)
    im = axes[0].imshow(mel, origin='lower')
    plt.colorbar(im, ax=axes[0])
    axes[1].plot(y)
    plt.tight_layout()
    plt.savefig('y.png', dpi=300)
    plt.close()
    wavfile.write('y.wav', 16000, y)
    return 

def mel_normalize(mel, mel_min=-12):
    return mel / (-mel_min / 2) + 1

def mel_random_masking(tensor, masking_ratio=0.1, mel_min=-12):

    mask = torch.rand(tensor.shape) > masking_ratio

    masked_tensor = torch.mul(tensor, mask)

    masked_tensor += ~mask * mel_min

    return masked_tensor

def mel_random_segment(mel, segment_len=250):
    M, T = mel.shape # (M, T)
    if T - segment_len > 0:
        random_offset = torch.randint(0, T - segment_len, (1,))
    else:
        random_offset = 0
    return mel[:, random_offset:random_offset+segment_len]

def batch_collator(metadata, masking_ratio=0.1, mel_min=-12):

    mel_list = list()
    emo_list = list()
    for f_file, script, emo in metadata:
        emo_list.append(EMO_DICT[emo])
        y = load_raw(f_file)
        mel = MEL2SAMP.get_mel(torch.tensor(y)) # (M, T)
        mel = mel_random_masking(mel, masking_ratio, mel_min)
        mel = mel_normalize(mel)
        mel = mel_random_segment(mel)
        mel_list.append(mel.T)
    
    batched_emo = torch.tensor(emo_list)
    batched_mel = pad_sequence(mel_list, batch_first=True) # (B, T, M)
    batched_mel = batched_mel.unsqueeze(1) # (B, 1, T, M)

    return batched_mel, batched_emo

class ReferenceEncoder(nn.Module):

    def __init__(self, conv_h, gru_h, num_emo):
        super(ReferenceEncoder, self).__init__()

        module_container = list()
        for h_in, h_out in zip([1] + conv_h[:-1], conv_h):
            module_container.extend([nn.Conv2d(h_in, h_out, kernel_size=3, stride=2, padding=1), 
                                     nn.ReLU(),
                                     nn.BatchNorm2d(h_out)])

        self.conv_layers = nn.Sequential(*module_container)
        
        self.gru = nn.GRU(conv_h[-1] * 2, gru_h, batch_first=True)
        # input of shape (batch, seq_len, input_size)

        self.fc = nn.Linear(gru_h, num_emo)

    def forward(self, input_tensor):
        # tensor = self.conv(input_tensor)
        # tensor = F.relu(tensor)
        # tensor = self.batch_norm(tensor)
        tensor = self.conv_layers(input_tensor) # (B, H, T, M) 
        tensor = tensor.transpose(1, 2) # (B, T, H, M[2]) 

        B, T, H, M = tensor.shape
        tensor = tensor.reshape(B, T, -1) # (B, T, H * M) 
 
        tensor, h_n = self.gru(tensor) # (L, N, H), (1, N, H)
        h_n.squeeze_(0) # (1, N, H) => (N, H)

        h_n = torch.tanh(h_n)

        pred = F.log_softmax(self.fc(h_n), dim=-1)

        return pred


'''
For the reference encoder architecture (Figure 2), 
we use a simple 6-layer convolutional network. 
Each layer is composed of 3 × 3 filters with 2 × 2 stride, 
SAME padding and ReLU activation. 
Batch normalization (Ioffe & Szegedy, 2015) is applied to every layer. 
The number of filters in each layer doubles at half the rate of downsampling: 
32, 32, 64, 64, 128,
'''

'''
We train our models for at least 200k steps with a minibatch size of 256 
using the Adam optimizer (Kingma & Ba, 2015). 
We start with a learning rate of 1 × 10−3 and decay it 
to 5 × 10−4, 3 × 10−4, 1 × 10−4, and 5 × 10−5 
at step 50k, 100k, 150k, and 200k respectively.
'''

if __name__ == "__main__":
    BASE_PATH = 'KoreanEmotionSpeech'
    metadata = load_meta(BASE_PATH)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(random.choice(metadata))

    # display_meta_statistics(metadata)

    y = load_raw(random.choice(metadata)[0])
    mel = MEL2SAMP.get_mel(torch.tensor(y))

    save_wav_plot(mel, y)

    json_dict = json.load(open('config.json'))
    model_params_dict = json_dict['model_config']
    hyper_params_dict = json_dict['training_config']

    if not os.path.isfile('train.csv') or not os.path.isfile('test.csv'):
        random.shuffle(metadata)
        split_idx = int(len(metadata) * hyper_params_dict['test_ratio'])
        test_meta = metadata[:split_idx]
        train_meta = metadata[split_idx:]

        with open('train.csv', 'w') as f:
            csv_writer =  csv.writer(f)
            csv_writer.writerows(train_meta)
        
        with open('test.csv', 'w') as f:
            csv_writer =  csv.writer(f)
            csv_writer.writerows(test_meta)
    else:
        with open('train.csv', 'r') as f:
            train_meta = [line for line in csv.reader(f)]
        with open('test.csv', 'r') as f:
            test_meta = [line for line in csv.reader(f)]
    
    print(model_params_dict)

    model = ReferenceEncoder(**model_params_dict)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=hyper_params_dict['lr'])
    loss_func = nn.NLLLoss()

    summary_writer = SummaryWriter()

    train_data_loader = DataLoader(train_meta, batch_size=hyper_params_dict['batch_size'], 
                                   shuffle=True, num_workers=4, 
                                   collate_fn=batch_collator, drop_last=False)

    test_data_loader = DataLoader(test_meta, batch_size=hyper_params_dict['batch_size'], 
                                  shuffle=False, num_workers=4, 
                                  collate_fn=batch_collator, drop_last=True)

    # input_tensor = torch.rand([4, 72, 80]).unsqueeze(1)

    step = 0

    for i in range(hyper_params_dict['num_epoch']):
        model.train()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo in tqdm(train_data_loader):
            optimizer.zero_grad()

            with autocast():
                pred_tensor = model(batched_mel.to(device))
                loss = loss_func(pred_tensor, batched_emo.to(device))

            # pred_tensor = model(batched_mel.to(device))
            # loss = loss_func(pred_tensor, batched_emo.to(device))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))
            step += 1

        summary_writer.add_scalar('loss/train', 
                                  np.mean(loss_list), 
                                  global_step=step)
        summary_writer.add_scalar('acc/train', 
                                  np.mean(acc_list), 
                                  global_step=step)

        print(f'[Train] Loss: {np.mean(loss_list):2.3f} / Acc: {np.mean(acc_list):2.3f}')

        model.eval()
        loss_list = list()
        acc_list = list()
        for batched_mel, batched_emo in tqdm(test_data_loader):
            
            with autocast():
                pred_tensor = model(batched_mel.to(device))
                loss = loss_func(pred_tensor, batched_emo.to(device))

            # pred_tensor = model(batched_mel.to(device))
            # loss = loss_func(pred_tensor, batched_emo.to(device))

            loss_list.append(loss.item())
            acc_list.append(torch.mean((torch.argmax(pred_tensor.cpu(), dim=-1) == batched_emo).float()))
        # print(np.mean(loss_list))
        summary_writer.add_scalar('loss/eval', 
                            np.mean(loss_list), 
                            global_step=step)
        summary_writer.add_scalar('acc/eval', 
                            np.mean(acc_list), 
                            global_step=step)

        print(f'[Eval] Loss: {np.mean(loss_list):2.3f} / Acc: {np.mean(acc_list):2.3f}')


    # pred_tensor = model(input_tensor)

    # print(pred_tensor)

    # print(f'{input_tensor.shape} => {pred_tensor.shape}')
