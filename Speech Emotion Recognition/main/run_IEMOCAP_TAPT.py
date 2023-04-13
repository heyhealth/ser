import os
import sys

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import torch
from torch.utils.tensorboard import SummaryWriter
from models.wav2vec2_wrappers import Wav2Vec2ForPreTrainingWrapper
from main.opts import ARGS
from utils.set_random_seed import setup_seed
from main.dataset import load_IEMOCAP_Dataset_for_TAPT
from main.train import train_model_TAPT, train_model_TAPT_on_multi_gpu, train_model_TAPT_on_distributed_mechine

setup_seed(ARGS.SEED)

summary_writer = SummaryWriter(log_dir='../save/runs/TAPT', flush_secs=120)

# data
train_iter, _ = load_IEMOCAP_Dataset_for_TAPT(batch_size=ARGS.TAPT_BATCH_SIZE)

# model
model = Wav2Vec2ForPreTrainingWrapper()

# task adaptive training
model = train_model_TAPT(model, train_iter, ARGS.TAPT_EPOCHS, ARGS.TAPT_WARMUP_STEPS,
                         ARGS.TAPT_LR, ARGS.TAPT_DEVICE,
                         writer=summary_writer)

# save model
torch.save(model, f'../save/checkpoints/TAPT.pt')
