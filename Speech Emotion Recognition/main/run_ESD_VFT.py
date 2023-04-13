import sys
import os

import torch

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training, train_model_for_aamsoftmax, \
    evaluate_model_when_finish_training_for_aamsoftmax
# from models.model_VFT import Wav2Vec2EmoNet
# from models.model_bare import Wav2Vec2EmoNet
# from models.model_VFT_v2 import Wav2Vec2EmoNet
# from models.model_VFT_v2_1 import Wav2Vec2EmoNet
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet
# from models.model_VFT_Res import Wav2Vec2EmoNet
# from models.model_VFT_v2 import WavLMEmoNet
# from models.model_VFT_Weights_Layers import Wav2Vec2EmoNet
from dataset import load_esd_basic_split
from utils.set_random_seed import setup_seed
import time

setup_seed(ARGS.SEED)

# data
train_iter, eval_iter, test_iter = load_esd_basic_split(batch_size=ARGS.BATCH_SIZE)

# model
net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_ESD)
# train
net = train_model(net, train_iter, test_iter=eval_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR,
                  weight_decay=ARGS.WEIGHT_DECAY,
                  device=ARGS.DEVICE, animator_name=f"ESD_train_{time.strftime('%m_%d_%H_%M')}", class_weight=None
                  )

# evaluate
_, WA, _, _, _, _, _ = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                                           cm_name=f"ESD_cm_{time.strftime('%m_%d_%H_%M')}",
                                                           display_labels=ARGS.CM_CLASS_LABELS_ESD)

# save model
torch.save(net.state_dict(),
           os.path.join(ARGS.MODEL_CHECKPOINTS_PATH, "ESD_SD_0228",
                        "ESD_model_params_WA_{:0.3f}.pt".format(WA)))
