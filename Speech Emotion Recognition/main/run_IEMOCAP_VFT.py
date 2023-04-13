import sys
import os

import torch

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training
# from models.model_VFT import Wav2Vec2EmoNet
# from models.model_bare import Wav2Vec2EmoNet
# from models.model_VFT_v2 import Wav2Vec2EmoNet
# from models.model_VFT_v2_1 import Wav2Vec2EmoNet
# from models.model_VFT_Res import Wav2Vec2EmoNet
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet,WavLMEmoNet
from dataset import load_iemocap_random_split
from utils.set_random_seed import setup_seed
import time

"""
Experiments:

    Speaker-Dependent
    IEMOCAP datasets
    train:test = 8:2

"""

setup_seed(ARGS.SEED)

# data
# random split the data
train_iter, test_iter, class_weights = load_iemocap_random_split(batch_size=ARGS.BATCH_SIZE,
                                                                 train_split_ratio=ARGS.TRAIN_SPLIT_RATIO)

# model wav2vec2
net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
# model wavlm
# net = WavLMEmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
# train
net = train_model(net, train_iter, test_iter=test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR,
                  weight_decay=ARGS.WEIGHT_DECAY,
                  device=ARGS.DEVICE, animator_name=f"IEMOCAP_SD_train_{time.strftime('%m_%d_%H_%M')}",
                  class_weight=class_weights)
# evaluate
res = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                          cm_name=f"IEMOCAP_SD_cm_{time.strftime('%m_%d_%H_%M')}",
                                          display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
# save
torch.save(net.state_dict(),
           os.path.join(ARGS.PROJECTION_PATH, 'save', 'checkpoints',
                        '{}_random_split_ua_{:0.3f}.pt'.format('wav2vec2_v2', res[0])))
