import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from train import compute_the_average_results, train_model, evaluate_model_when_finish_training
from models.MultiLevel_Model0 import Wav2Vec2EmoNet
from dataset import load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out
from utils.set_random_seed import setup_seed

"""
5-fold cross validate
leave-one-session-out
"""

setup_seed(ARGS.SEED)

all_fold_results = []
for fold in range(ARGS.FOLD):
    # data
    train_iter, test_iter, class_weight = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out(
        session_index=(fold + 1),
        batch_size=ARGS.BATCH_SIZE)
    # model
    net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
    # train
    net = train_model(net, train_iter, test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY,
                      device=ARGS.DEVICE, animator_name=f"leave_one_session_out_{(fold + 1)}",
                      class_weight=class_weight)
    # evaluate
    results = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                                  cm_name=f"cm_leave_one_session_out_{(fold + 1)}",
                                                  display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
    all_fold_results.append(results)

compute_the_average_results(all_fold_results)
