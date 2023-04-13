import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from train import train_model, evaluate_model_when_finish_training
from models.VanillaFineTuningModels1 import Wav2Vec2EmoNet
from dataset import load_RAVDESS_Dataset
from utils.set_random_seed import setup_seed
import time

setup_seed(ARGS.SEED)

# data
train_iter, test_iter, class_weight = load_RAVDESS_Dataset(batch_size=ARGS.BATCH_SIZE)
# model
net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_RAVDESS)
# train
net = train_model(net, train_iter, test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY,
                  device=ARGS.DEVICE, animator_name=f"RAVDESS_train_{time.strftime('%m_%d_%H_%M')}",
                  class_weight=class_weight)
# evaluate
evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                    cm_name=f"RAVDESS_cm_{time.strftime('%m_%d_%H_%M')}",
                                    display_labels=ARGS.CM_CLASS_LABELS_RAVDESS)
