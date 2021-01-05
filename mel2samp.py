import sys
sys.path.insert(0, 'waveglow')
sys.path.insert(0, 'waveglow/tacotron2')
# print(sys.path)
from waveglow.mel2samp import Mel2Samp

import json

mel_dict = json.load(open('config.json'))['mel_config']
MEL2SAMP = Mel2Samp(**mel_dict)

if __name__ == "__main__":
    mel_dict = json.load(open('config.json'))['mel_config']
    print(Mel2Samp)
    mel2samp = Mel2Samp(**mel_dict)