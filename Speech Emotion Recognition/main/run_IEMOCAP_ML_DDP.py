import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
from opts import ARGS
from main.train import train_model, evaluate_model_when_finish_training, train_model_on_distributed_mechine
from models.MultiLevel_Model0 import Wav2Vec2EmoNet
from dataset import load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out, \
    load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out_DDP
from utils.set_random_seed import setup_seed
import time
from torch import distributed
import torch


setup_seed(ARGS.SEED)

# init ddp
distributed.init_process_group(backend='nccl')
local_rank = distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# data
# random split the data
# train_iter, test_iter = load_IEMOCAP_Dataset(batch_size=ARGS.BATCH_SIZE, split_ratio=ARGS.SPLIT_RATIO)
# leave one session out (speaker independent)
train_iter, test_iter, class_weight = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out_DDP(session_index=5,
                                                                                                   batch_size=ARGS.BATCH_SIZE)

_, test_iter_, _ = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out(session_index=5, batch_size=ARGS.BATCH_SIZE)

# model
net = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
# train (DDP)
net = train_model_on_distributed_mechine(net, train_iter, test_iter, epochs=ARGS.EPOCHS, lr=ARGS.LR,
                                         weight_decay=ARGS.WEIGHT_DECAY,local_rank=local_rank,
                                         device=device, animator_name=f"IEMOCAP_train_{time.strftime('%m_%d_%H_%M')}",
                                         class_weight=class_weight)
# 在一块设备上进行最后的模型评估
if local_rank==0:
    # save model
    net_ = Wav2Vec2EmoNet(num_classes=ARGS.NUM_CLASSES_IEMOCAP)
    net_.load_state_dict(net.module.state_dict())
    # evaluate
    evaluate_model_when_finish_training(net_, test_iter_, device=device,
                                        cm_name=f"IEMOCAP_cm_{time.strftime('%m_%d_%H_%M')}",
                                        display_labels=ARGS.CM_CLASS_LABELS_IEMOCAP)
