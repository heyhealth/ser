import sys
import os

import torch

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training, train_model_for_aamsoftmax, \
    evaluate_model_when_finish_training_for_aamsoftmax, compute_the_average_results
# from models.model_VFT import Wav2Vec2EmoNet
# from models.model_bare import Wav2Vec2EmoNet
# from models.model_VFT_v2 import Wav2Vec2EmoNet
# from models.model_VFT_v2_1 import Wav2Vec2EmoNet
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet
# from models.model_VFT_Res import Wav2Vec2EmoNet
# from models.model_VFT_v2 import WavLMEmoNet
# from models.model_VFT_Weights_Layers import Wav2Vec2EmoNet
from dataset import load_esd_basic_split, load_esd_one_speaker_with_eval, load_esd_concat_iemocap_one_speaker_with_eval
from utils.set_random_seed import setup_seed
import time

"""
Experiments:

    Speaker-Independent (10-fold cross validate;leave-one-speaker-out)
    data:  train_iter(iemocap+esd),eval_iter(esd),test_iter(esd)
    model: wav2vec2_v3: cls_token
    early stopping:  (train_acc >= 0.950 and train_loss <= 0.150) or (test_acc >= 0.9 if test_acc is not None else False) 85%
    

"""

setup_seed(ARGS.SEED)

# CM_CLASS_LABELS_ESD = ARGS.CM_CLASS_LABELS_ESD
CM_CLASS_LABELS_ESD = ["Angry", "Happy", "Sad", "Neutral"]

EVERY_FOLD_RESULTS_LIST = []

for speaker_index in range(10):
    test_speaker_index = "{:04d}".format(speaker_index + 1)
    eval_speaker_index = "{:04d}".format((speaker_index + 2) if speaker_index != 9 else 1)
    """
    # basic data
    train_iter, eval_iter, test_iter = load_esd_one_speaker_with_eval(eval_speaker_index, test_speaker_index,
                                                                      ARGS.BATCH_SIZE)
    """
    # fusion iemocap and esd data
    train_iter, eval_iter, test_iter, class_weight = load_esd_concat_iemocap_one_speaker_with_eval(eval_speaker_index, test_speaker_index,
                                                                             ARGS.BATCH_SIZE)
    # model
    # net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_ESD)
    net = Wav2Vec2EmoNet(num_classes=4)
    # train
    net = train_model(net, train_iter, test_iter=eval_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR,
                      weight_decay=ARGS.WEIGHT_DECAY,
                      device=ARGS.DEVICE, animator_name=f"train_loso_speaker_{eval_speaker_index}",
                      class_weight=class_weight
                      )

    # evaluate
    results = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                                  cm_name=f"cm_loso_speaker_{test_speaker_index}",
                                                  display_labels=CM_CLASS_LABELS_ESD)
    # save model: !!!config the save path!!!
    """
    torch.save(net.state_dict(), os.path.join(ARGS.MODEL_CHECKPOINTS_PATH, "ESD_LOSO_DATA_FUSION_Wav2Vec2_V3_class4",
                                              'ESD_model_params_{}_loso_speaker_{}_ua_{:0.3f}.pt'.format("wav2vec2_v3",
                                                  test_speaker_index,
                                                  results[1])))
    """
    torch.save(net.state_dict(), os.path.join(ARGS.PROJECTION_PATH, "save", "checkpoints",
                                              'ESD_model_params_{}_loso_speaker_{}_ua_{:0.3f}.pt'.format("wav2vec2_v3",
                                                                                                         test_speaker_index,
                                                                                                         results[1])))
    # log
    print("--------------------------", f"The Fold {speaker_index + 1} Has Done~", "--------------------------",
          sep='\n',end='\n\ns')
    EVERY_FOLD_RESULTS_LIST.append(results)

compute_the_average_results(EVERY_FOLD_RESULTS_LIST, display_labels=CM_CLASS_LABELS_ESD)
