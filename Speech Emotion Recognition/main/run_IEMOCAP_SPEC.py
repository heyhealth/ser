import os, sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))

from main.dataset import load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out
from models.DeepLearningModels import EmoNet_ResNet101, EmoNet_ResNet34, EmoNet_ResNet18, EmoNet_AlexNet, \
    EmoNet_EfficientNet, EmoNet_SSAST
from utils.set_random_seed import setup_seed
from main.opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training
import time

setup_seed(ARGS.SEED)

# data
train_iter, test_iter, class_weight = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out(session_index=5,
                                                                                               batch_size=ARGS.BATCH_SIZE)
# model
net = EmoNet_SSAST(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
# train
net = train_model(net, train_iter, test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY,
                  device=ARGS.DEVICE, animator_name=f"IEMOCAP_train_{time.strftime('%m_%d_%H_%M')}",
                  class_weight=class_weight)
# evaluate
evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                    cm_name=f"IEMOCAP_cm_{time.strftime('%m_%d_%H_%M')}",
                                    display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
