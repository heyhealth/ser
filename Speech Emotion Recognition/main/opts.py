import argparse
import time
import torch
import os
import sys

Parser = argparse.ArgumentParser(description='the Speech Emotion Recognition needed parameters setup')
# -------------------------- Start --------------------------

# -------------------------- OS --------------------------
SYS = 'linux' if sys.platform == 'linux' else 'windows'
assert SYS in ['linux', 'windows']

Parser.add_argument('--PROJECTION_PATH',
                    default=os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))

# -------------------------- data --------------------------
Parser.add_argument('--TRAIN_SPLIT_RATIO', default=0.8, help='the ratio of train / total data')
Parser.add_argument('--AUDIO_SEGMENT_TIME_LENGTH', type=float, default=5,
                    help='the args of Wav2Vec2 Config,suggest the value is same with the length of audio')

# -------------------------- IEMOCAP dataset --------------------------
Parser.add_argument('--IEMOCAP_DATASET_FOLDER',
                    default=r'D:\datasets\IEMOCAP' if SYS == 'windows' else '/home/user/health/datasets/IEMOCAP/raw_unzip')
# 10->7->4
Parser.add_argument('--IEMOCAP_SEQUENCE_LENGTH', default=5,
                    help='the max sequence length for IEMOCAP datasets (second)')
# ["ang(exc)":1103, "hap":1636, "sad":1084, "neu":1708]

Parser.add_argument('--FILTERS_IEMOCAP', default={
    "SESSION": [
        "Session1",
        "Session2",
        "Session3",
        "Session4",
        "Session5"
    ],
    "SPEAKER": [
        "Ses01F", "Ses01M",
        "Ses02F", "Ses02M",
        "Ses03F", "Ses03M",
        "Ses04F", "Ses04M",
        "Ses05F", "Ses05M",
    ],
    "EMOTION": [
        # "ang", "hap", "sad", "neu", "exc",
        "ang", "hap", "exc", "sad", "fru", "fea", "sur", "neu", "xxx"
    ]

}, help='filters of IEMOCAP dataset')

# -------------------------- RAVDESS dataset --------------------------

Parser.add_argument('--RAVDESS_DATASET_Modality_FOLDER',
                    default=r"E:\Datasets\RAVDESS\raw\Audio-only-files" if SYS == 'windows' else "/home/user/health/datasets/RAVDESS/raw/Audio-only-files")
Parser.add_argument('--RAVDESS_SEQUENCE_LENGTH', default=5,
                    help='the max sequence length for RAVDESS datasets (second)')
Parser.add_argument('--FILTERS_RAVDESS', default={
    'Modality': ['01', '02', '03'],
    'Vocal_channel': ['01', '02'],
    # (speech:1440 song:1012)
    'Emotion': ['01', '03', '04', '05', '06', '07', '08', '02'],
    # 2452= 564('01':188,'02':376)+376+376+376+376+192+192
    'Emotional_intensity': ['01', '02'],
    'Statement': ['01', '02'],
    'Repetition': ['01', '02'],
    'Actor': [f"{index:02d}" for index in range(1, 25)],
    'wav_path': None
}, help='filters of RAVDESS dataset')

# -------------------------- Emotional Speech Dataset (ESD) dataset --------------------------
Parser.add_argument('--ESD_DATASET_FOLDER',
                    default=r'D:\datasets\ESD' if SYS == 'windows' else '/home/user/health/datasets/ESD')
# 10
Parser.add_argument('--ESD_SEQUENCE_LENGTH', default=5,
                    help='the max sequence length for ESD datasets (second)')

Parser.add_argument('--FILTERS_ESD', default={
    "SPLIT": [
        "train",
        "test",
        "evaluation",
    ],
    "SPEAKER": [
        '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008', '0009', '0010',  # Chinese
        # '0011', '0012', '0013', '0014','0015', '0016', '0017', '0018', '0019', '0020' # English
    ],
    "EMOTION": [
        "Angry", "Happy", "Neutral", "Sad", "Surprise"
    ]

}, help='filters of Emotional Speech Dataset (ESD) dataset')

# -------------------------- Wav2Vec2 model --------------------------
Parser.add_argument('--MODEL_NAME_Wav2Vec2',
                    default=r"C:\Users\health\.cache\huggingface\hub\models--facebook--wav2vec2-base-960h\snapshots\706111756296bc76512407a11e69526cf4e22aae" if SYS == 'windows' else "/home/user/health/huggingface/hub/models--facebook--wav2vec2-base-960h/snapshots/706111756296bc76512407a11e69526cf4e22aae")
Parser.add_argument('--HIDDEN_STATES_DIM', type=int, default=768)

# -------------------------- WavLM model --------------------------
Parser.add_argument('--MODEL_NAME_WavLM',
                    default=r"C:\Users\health\.cache\huggingface\hub\models--patrickvonplaten--wavlm-libri-clean-100h-base-plus\snapshots\02c289c4471cd1ba4b0ff3e7c304afe395c5026a" if SYS == 'windows' else "/home/user/health/huggingface/hub/models--patrickvonplaten--wavlm-libri-clean-100h-base-plus/snapshots/02c289c4471cd1ba4b0ff3e7c304afe395c5026a")
# Parser.add_argument('--MODEL_NAME_WavLM',
#                     default=r"C:\Users\health\.cache\huggingface\hub\models\WavLM-Base+.pt" if SYS == 'windows' else "/home/user/health/huggingface/hub/models/WavLM-Base+.pt")

# -------------------------- SSAST model --------------------------
Parser.add_argument('--MODEL_NAME_SSAST',
                    default=r"C:\Users\health\.cache\huggingface\hub\models\SSAST-Tiny-Frame-400.pth" if SYS == 'windows' else "/home/user/health/huggingface/hub/models/SSAST-Tiny-Frame-400.pth")

# SSAST-Base-Frame-400.pth
# SSAST-Tiny-Frame-400.pth
# SSAST-Tiny-Patch-400.pth

# -------------------------- models  --------------------------

Parser.add_argument('--MODEL_CHECKPOINTS_PATH',
                    default="/home/user/health/reposity/SER/models_hub")

# -------------------------- model classes --------------------------

Parser.add_argument('--NUM_CLASSES_IEMOCAP', default=9)
Parser.add_argument('--NUM_CLASSES_RAVDESS', default=7)
Parser.add_argument('--NUM_CLASSES_ESD', default=5)

# -------------------------- train --------------------------
Parser.add_argument('--SEED', default=3407)
# Parser.add_argument('--SEED', default=int(time.time()))
Parser.add_argument('--FOLD', default=5)
Parser.add_argument('--EPOCHS', default=50)
Parser.add_argument('--BATCH_SIZE', default=32)
Parser.add_argument('--LR', default=1e-5)
Parser.add_argument('--LR_ASR', default=1e-5)
Parser.add_argument('--LR_SER', default=1e-6)
Parser.add_argument('--WEIGHT_DECAY', default=0)
Parser.add_argument('--DEVICE', default='cuda:6' if torch.cuda.is_available() else 'cpu')

# -------------------------- DataParallel --------------------------

Parser.add_argument('--NUM_GPUS', default=4, help='for DP')

# -------------------------- DistributedDataParallel --------------------------
"""
# 其中CUDA_VISIBLE_DEVICES指定机器上显卡的数量
# nproc_per_node程序进程的数量     
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py
"""
Parser.add_argument('--local_rank', type=int, help='for DDP')

# -------------------------- Wav2Vec2 Task adaptive training stage en
# en--------------------------

Parser.add_argument('--TAPT_LR', type=float, default=1e-6)
Parser.add_argument('--TAPT_BATCH_SIZE', type=int, default=20)
Parser.add_argument('--TAPT_EPOCHS', type=int, default=120, help='Default 120000')
Parser.add_argument('--TAPT_WARMUP_STEPS', type=int, default=4, help='Default 4000')
Parser.add_argument('--TAPT_DEVICE', default='cuda:3' if torch.cuda.is_available() else 'cpu')

# -------------------------- visualization --------------------------

Parser.add_argument('--CM_CLASS_LABELS_IEMOCAP',
                    default=["ang", "hap", "exc", "sad", "fru", "fea", "sur", "neu", "xxx"],
                    help='ConfusionMatrixDisplay name for IEMOCAP')
Parser.add_argument('--CM_CLASS_LABELS_RAVDESS',
                    default=["neutral(calm)", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
                    help='ConfusionMatrixDisplay name for RAVDESS')
Parser.add_argument('--CM_CLASS_LABELS_ESD', default=["Angry", "Happy", "Neutral", "Sad", "Surprise"],
                    help='ConfusionMatrixDisplay name for ESD')
# -------------------------- End --------------------------
ARGS = Parser.parse_args()
