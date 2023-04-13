import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from train import compute_the_average_results, train_model, evaluate_model_when_finish_training
# from models.model_VFT import Wav2Vec2EmoNet
# from models.model_VFT_v2 import Wav2Vec2EmoNet
from models.VanillaFineTuningModels2 import EmoNet_WavLM
from dataset import load_iemocap_one_speaker_without_eval
from utils.set_random_seed import setup_seed
import torch

"""
10-fold cross validate
leave-one-speaker-out
"""

setup_seed(ARGS.SEED)

all_fold_results = []
# 10 speaker LOSO
for speaker_index in range(10):
    # data
    train_iter, test_iter, class_weight = load_iemocap_one_speaker_without_eval(
        speaker_index=speaker_index,
        batch_size=ARGS.BATCH_SIZE)
    # model
    net = EmoNet_WavLM(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
    # train
    net = train_model(net, train_iter, test_iter=None, epochs=ARGS.EPOCHS, lr=ARGS.LR, weight_decay=ARGS.WEIGHT_DECAY,
                      device=ARGS.DEVICE, animator_name=f"train_loso_speaker_{(speaker_index + 1)}",
                      class_weight=class_weight)
    """
    # save model
    torch.save(net.static_dict(),
               os.path.join(ARGS.PROJECTION_PATH, 'save', 'checkpoints',
                            f"model_{'cls_token'}_LOSPO_fold_{speaker_index + 1}.pt"))
    # load trained model
    net = torch.load(
        os.path.join(ARGS.PROJECTION_PATH, 'save', 'checkpoints', f"model_{'cls_token'}_LOSEO_fold_{speaker_index + 1}.pt"))
    """
    # evaluate
    results = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                                  cm_name=f"cm_loso_speaker_{(speaker_index + 1)}",
                                                  display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
    # log
    print("--------------------------", f"The Fold {speaker_index + 1} Has Done~", "--------------------------",
          sep='\n')
    all_fold_results.append(results)

compute_the_average_results(all_fold_results)
