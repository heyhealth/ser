import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from train import compute_the_average_results, train_model_for_multi_task, evaluate_model_when_finish_training
from models.MultiTaskModels import Wav2Vec2EmoNet
from dataset import load_iemocap_one_session_multi_task
from utils.set_random_seed import setup_seed
from mydatasets.iemocap import get_vocal_size_IEMOCAP
import torch

"""
Experiments:

    Speaker-Independent (5-fold cross validate;leave-one-session-out)
    data: IEMOCAP datasets: train_iter,eval_iter,test_iter
    model: model_Multi_Task extends wav2vec2_v3: cls_token
    early stopping: train_acc >= 0.950 and train_loss <= 0.150:
    
"""
setup_seed(ARGS.SEED)

all_fold_results = []
for fold in range(5):
    # data
    test_session_index = (fold + 2) if fold != 4 else 1
    train_iter, eval_iter, test_iter, class_weight = load_iemocap_one_session_multi_task(
        eval_session_index=(fold + 1), test_session_index=test_session_index, batch_size=ARGS.BATCH_SIZE)
    # model
    net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP, vocab_size=get_vocal_size_IEMOCAP())
    # train
    net = train_model_for_multi_task(net, train_iter, test_iter=eval_iter, epochs=ARGS.EPOCHS, lr_asr=ARGS.LR_ASR,
                                     lr_ser=ARGS.LR_SER,
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
                            f"model_{'wav2vec2_multi_task'}_loso_session_{test_session_index}_ua_{'{:0.3f}'.format(results[0])}.pt"))

    # log
    print("--------------------------", f"The Fold {fold + 1} Has Done~", "--------------------------", sep='\n')
    all_fold_results.append(results)

compute_the_average_results(all_fold_results, ARGS.CM_CLASS_LABELS_IEMOCAP)
