import os
from d2l import torch as d2l
import torch
from torch.nn import functional as F
from main.opts import ARGS
from mydatasets.feature_acoustic import extract_waveform_from_wav
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet
from sklearn.preprocessing import MinMaxScaler

EMO_LAB_MAP_DIC = dict(zip(
    ARGS.FILTERS_ESD['EMOTION'], list(range(len(ARGS.FILTERS_ESD['EMOTION'])))
))

# default reasoning on cpu device
DEVICE = 'cpu'
# DEVICE = 'cuda:5'

timer = d2l.Timer()
timer.start()
# load the audio
audio_extracted = extract_waveform_from_wav(os.path.join(ARGS.PROJECTION_PATH, 'save', 'data', '0004_000752.wav'),
                                            max_sequence_length=4)
# to tensor
audio_input_values, audio_valid_length = torch.as_tensor(audio_extracted[0], dtype=torch.float32).unsqueeze(
    0).to(DEVICE), torch.as_tensor(
    audio_extracted[1],
    dtype=torch.long).unsqueeze(0).to(DEVICE)

# load the model
with torch.no_grad():
    """
    model = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_ESD)
    reasoning_model = torch.load(ARGS.REASONING_MODEL)
    model.load_state_dict(reasoning_model.state_dict())  # load the pretrained parameters
    """
    model = torch.load(ARGS.REASONING_MODEL).to(DEVICE)

    # outputs the logits
    logits = model(audio_input_values, audio_valid_length)

# print the logits
print("--------------------------------------")
for label, prob in zip(EMO_LAB_MAP_DIC.keys(), logits.squeeze().cpu().numpy()):
    if EMO_LAB_MAP_DIC.get(label) == torch.argmax(logits, dim=1).numpy():
        print("{}:{:0.3f}  √".format(label, prob))
        continue
    print("{}:{:0.3f}".format(label, prob))
print("--------------------------------------")
# print the consumptive time
timer.stop()
print("total cost time(s):{}".format(timer.sum()))
