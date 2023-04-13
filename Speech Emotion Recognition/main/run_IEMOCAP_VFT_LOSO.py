import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from train import compute_the_average_results, train_model, evaluate_model_when_finish_training
# from models.model_VFT import Wav2Vec2EmoNet
# from models.model_VFT_v2 import Wav2Vec2EmoNet
# from models.model_VFT_v2 import WavLMEmoNet
from models.VanillaFineTuningModels4 import Wav2Vec2EmoNet,WavLMEmoNet
# from models.model_VFT_Weights_Layers import Wav2Vec2EmoNet
from dataset import load_iemocap_one_session_with_eval, load_iemocap_one_session_without_eval, \
    load_iemocap_concat_esd_one_session_with_eval
from utils.set_random_seed import setup_seed
import torch

"""
Experiments:

    Speaker-Independent (5-fold cross validate;leave-one-session-out)
    data: IEMOCAP datasets: train_iter,eval_iter,test_iter
    model: wav2vec2_v3: cls_token
    early stopping: (train_acc >= 0.900 and train_loss <= 0.150) or (test_acc >= 0.8 if test_acc is not None else False)
    
"""

setup_seed(ARGS.SEED)

all_fold_results = []
for fold in range(5):
    # data (loso without eavl_iter)
    """
    test_session_index = (fold + 2) if fold != 4 else 1
    train_iter, eval_iter, test_iter, class_weight = load_iemocap_one_session_with_eval(
        eval_session_index=(fold + 1), test_session_index=test_session_index, batch_size=ARGS.BATCH_SIZE)
    """
    # data fusion (ESD->IEMOCAP)
    """
    train_iter, eval_iter, test_iter, class_weight = load_iemocap_concat_esd_one_session_with_eval(
        eval_session_index=(fold + 1), test_session_index=test_session_index, batch_size=ARGS.BATCH_SIZE)
    """
    # data (loso without eavl_iter)
    test_session_index = fold + 1
    train_iter, test_iter, class_weight = load_iemocap_one_session_without_eval(
        session_index=test_session_index, batch_size=ARGS.BATCH_SIZE
    )
    eval_iter = test_iter

    # model
    # net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
    net = WavLMEmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
    # train
    net = train_model(net, train_iter, test_iter=eval_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR,
                      weight_decay=ARGS.WEIGHT_DECAY,
                      device=ARGS.DEVICE, animator_name=f"train_loso_session_{test_session_index}",
                      class_weight=class_weight)

    # evaluate
    results = evaluate_model_when_finish_training(net, test_iter, device=ARGS.DEVICE,
                                                  cm_name=f"cm_loso_session_{test_session_index}",
                                                  display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
    # save model
    torch.save(net.state_dict(),
               os.path.join(ARGS.PROJECTION_PATH, 'save', 'checkpoints',
                            f"model_{'WavLM'}_loso_session_{test_session_index}_ua_{'{:0.3f}'.format(results[0])}.pt"))
    # log
    print("--------------------------", f"The Fold {fold + 1} Has Done~", "--------------------------", sep='\n')
    all_fold_results.append(results)

compute_the_average_results(all_fold_results, ARGS.CM_CLASS_LABELS_IEMOCAP)
