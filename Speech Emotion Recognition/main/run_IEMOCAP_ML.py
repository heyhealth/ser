import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training,train_model_on_multi_gpu
from models.MultiLevel_Model0 import Wav2Vec2EmoNet
from dataset import load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out
from utils.set_random_seed import setup_seed
import time

setup_seed(ARGS.SEED)

# data
# random split the data
# train_iter, test_iter = load_IEMOCAP_Dataset(batch_size=ARGS.BATCH_SIZE, split_ratio=ARGS.SPLIT_RATIO)
# leave one session out (speaker independent)
train_iter, test_iter, class_weight = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out(session_index=5,
                                                                                               batch_size=ARGS.BATCH_SIZE)
# model
net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)

# train (以DDP方式单机多卡训练)
net = train_model(net, train_iter, test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY,
                  device=ARGS.DEVICE, animator_name=f"IEMOCAP_train_{time.strftime('%m_%d_%H_%M')}",
                  class_weight=class_weight)

# evaluate
evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                    cm_name=f"IEMOCAP_cm_{time.strftime('%m_%d_%H_%M')}",
                                    display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
