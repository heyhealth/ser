"""
The model pretrained on IEMOCAP transfer to ESD without fine-tunning
"""

"""
IEMOCAP:
 "EMOTION": [
        "ang", "hap", "sad", "neu", "exc"
    ]

ESD:
  "EMOTION": [
        "Angry", "Happy", "Neutral", "Sad", "Surprise"
    ]

1. Ignore the 'Surprise' in ESD data
2. Switch the order (Neutral-Sad)
"""
import torch
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet
from main.dataset import load_esd_total
from main.train import evaluate_model_transfer_learning

# set device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

# load model
IEMOCAP_MODEL_WAV2VEC2_V3_PATH = r"E:\A_Experiments\IEMOCAP\random_split\wav2vec2_v3\wav2vec2_v3_random_split_ua_0.759"

net = Wav2Vec2EmoNet(num_classes=4)

# net.load_state_dict(torch.load(IEMOCAP_MODEL_WAV2VEC2_V3_PATH))
# load data
data_iter = load_esd_total(batch_size=12, filtered_emotion=["Angry", "Happy", "Sad", "Neutral"],filtered_speaker=['0001'])

# model reasoning
results = evaluate_model_transfer_learning(net, data_iter, device, cm_name='iemocap2esd_speaker_0001_cm',
                                           display_labels=["ang", "hap", "sad", "neu"])
