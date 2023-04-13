import sys
import os
import torch

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import copy
import numpy as np
from torch.utils import data
from sklearn.model_selection import StratifiedKFold
from mydatasets.iemocap import IEMOCAP, IEMOCAP_A_T_PAIR
from mydatasets.esd import ESDDataset
from mydatasets.ravdess import RAVDESS
from utils.data_related import Dataset_
from main.opts import ARGS


def load_IEMOCAP_X_Y_arrays():
    """
    load X-array,y-array IEMOCAP DATASETS
    :return: X_array, Y_array
    """
    X_array = []
    Y_array = []
    iemocap = IEMOCAP()
    class_weights = iemocap._get_class_weight()

    for (x, length), y in iemocap:
        X_array.append(x.cpu().numpy())
        Y_array.append(y.cpu().numpy())
    return np.array(X_array), np.array(Y_array), class_weights


def load_iemocap_random_split(batch_size=4, train_split_ratio=0.8):
    iemocap = IEMOCAP()
    class_weight = iemocap._get_class_weight()
    train_data, test_data = data.random_split(iemocap, [train_split_ratio, 1 - train_split_ratio])

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True), data.DataLoader(test_data,
                                                                                                              batch_size=batch_size,
                                                                                                              shuffle=False)
    return train_iter, test_iter, class_weight


def load_iemocap_dataset_sd(batch_size=4, split_ratio=0.8):
    """
    load IEMOCAP DATASETS (speaker-dependent)(sd)
    :param batch_size:
    :param split_ratio:
    :return: train_iter,test_iter with batch
    """
    X, y, class_weights = load_IEMOCAP_X_Y_arrays()
    # for the data imbalanced
    skf = StratifiedKFold(int(1 / (1 - split_ratio)))
    for train_index, test_index in skf.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    train_data, test_data = Dataset_(X_train, y_train), Dataset_(X_test, y_test)
    # random split
    """
    length = int(len(iemocap) * split_ratio)
    train_data, test_data = data.random_split(iemocap, lengths=[length, len(iemocap) - length],
                                              generator=torch.Generator().manual_seed(2022))
    """

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True), data.DataLoader(test_data,
                                                                                                              batch_size=batch_size,
                                                                                                              shuffle=False)

    return train_iter, test_iter, class_weights


def load_iemocap_one_session_without_eval(session_index, batch_size=4):
    """
    load IEMOCAP datasets for 5 fold cross validate, leave one session to test (without eval_iter)
        one of the most used split style
    :param batch_size:
    :param session_index: the session which is leave
    :return: train_iter,test_iter with batch
    """
    SESSIONS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SESSION'])
    SESSIONS.remove(f'Session{session_index}')
    train_data = IEMOCAP(filtered_session=SESSIONS)
    class_weight = train_data._get_class_weight()
    test_data = IEMOCAP(filtered_session=[f'Session{session_index}'])

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True), data.DataLoader(test_data,
                                                                                                              batch_size=batch_size,
                                                                                                              shuffle=False)

    return train_iter, test_iter, class_weight


def load_iemocap_one_session_with_eval(eval_session_index, test_session_index, batch_size=4):
    """
    load the IEMOCAP datasets
    :param eval_session_index:
    :param test_session_index:
    :param batch_size:
    :return: train_iter,eval_iter,test_iter,class_weight
    """
    SESSIONS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SESSION'])
    SESSIONS.remove("Session{}".format(eval_session_index))
    SESSIONS.remove("Session{}".format(test_session_index))
    train_data = IEMOCAP(filtered_session=SESSIONS)
    class_weight = train_data._get_class_weight()
    eval_data = IEMOCAP(filtered_session=["Session{}".format(eval_session_index)])
    test_data = IEMOCAP(filtered_session=["Session{}".format(test_session_index)])

    train_iter, eval_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size,
                                                       shuffle=True), data.DataLoader(eval_data,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False), data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    return train_iter, eval_iter, test_iter, class_weight


def load_iemocap_one_session_multi_task(eval_session_index, test_session_index, batch_size=4):
    """
    load the IEMOCAP datasets
    :param eval_session_index:
    :param test_session_index:
    :param batch_size:
    :return: train_iter,eval_iter,test_iter,class_weight
    """
    SESSIONS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SESSION'])
    SESSIONS.remove("Session{}".format(eval_session_index))
    SESSIONS.remove("Session{}".format(test_session_index))
    train_data = IEMOCAP_A_T_PAIR(filtered_session=SESSIONS, needed_text=True)
    class_weight = train_data._get_class_weight()
    eval_data = IEMOCAP_A_T_PAIR(filtered_session=["Session{}".format(eval_session_index)])
    test_data = IEMOCAP_A_T_PAIR(filtered_session=["Session{}".format(test_session_index)])

    train_iter, eval_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size,
                                                       shuffle=True), data.DataLoader(eval_data,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False), data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    return train_iter, eval_iter, test_iter, class_weight


def load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out_DDP(session_index, batch_size=4):
    """
    load IEMOCAP datasets for 5 fold cross validate, leave one session to test
    :param batch_size:
    :param session_index: the session which is leave
    :return: train_iter,test_iter with batch
    """
    SESSIONS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SESSION'])
    SESSIONS.remove(f'Session{session_index}')
    train_data = IEMOCAP(filtered_session=SESSIONS)
    class_weight = train_data._get_class_weight()
    test_data = IEMOCAP(filtered_session=[f'Session{session_index}'])

    train_sampler = data.distributed.DistributedSampler(train_data)
    test_sampler = data.distributed.DistributedSampler(test_data)

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size,
                                            sampler=train_sampler), data.DataLoader(test_data, batch_size=batch_size,
                                                                                    sampler=test_sampler)

    return train_iter, test_iter, class_weight


def load_IEMOCAP_Dataset_for_TAPT(batch_size=4, split_ratio=1.0):
    """
    load the dataset from task adaptive pretraining state
    :param batch_size:
    :param split_ratio: the train data / the all data
    :return: train_data , None
    """
    iemocap = IEMOCAP()
    num_training = int(len(iemocap) * split_ratio)
    splits = [num_training, len(iemocap) - num_training]

    train_data, test_data = data.random_split(iemocap, splits, generator=torch.Generator())

    return data.DataLoader(train_data, batch_size=batch_size, shuffle=True), None


def load_IEMOCAP_Dataset_for_TAPT_DDP(batch_size=4, split_ratio=1.0):
    """
    load the dataset from task adaptive pretraining state
    :param batch_size:
    :param split_ratio: the train data / the all data
    :return: train_data , None
    """
    iemocap = IEMOCAP()
    num_training = int(len(iemocap) * split_ratio)
    splits = [num_training, len(iemocap) - num_training]

    train_data, test_data = data.random_split(iemocap, splits, generator=torch.Generator())

    train_data_sampler = data.distributed.DistributedSampler(train_data, shuffle=True)

    return data.DataLoader(train_data, batch_size=batch_size, sampler=train_data_sampler), None


def load_RAVDESS_Dataset(batch_size=4, split_ratio=None):
    filters_train = ARGS.FILTERS
    filters_train.update({'Actor': [f'{index:02d}' for index in list(range(1, 23))]})
    filters_test = copy.deepcopy(ARGS.FILTERS)
    filters_test.update({'Actor': [f'{index:02d}' for index in list(range(23, 25))]})
    train_data = RAVDESS(filters=filters_train)
    class_weigth = train_data._get_class_weight()
    test_data = RAVDESS(filters=filters_test)

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True), data.DataLoader(test_data,
                                                                                                              batch_size=batch_size,
                                                                                                              shuffle=False)
    return train_iter, test_iter, class_weigth


def load_iemocap_one_speaker_without_eval(speaker_index, batch_size=4):
    SPEAKERS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SPEAKER'])
    SPEAKERS.pop(speaker_index)
    train_data = IEMOCAP(filtered_speaker=SPEAKERS)
    class_weight = train_data._get_class_weight()
    test_data = IEMOCAP(filtered_speaker=[ARGS.FILTERS_IEMOCAP['SPEAKER'][speaker_index]])

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size, shuffle=True), data.DataLoader(test_data,
                                                                                                              batch_size=batch_size,
                                                                                                              shuffle=False)

    return train_iter, test_iter, class_weight


def load_IEMOCAP_for_LOSO_10_fold_for_DDP(speaker_index, batch_size=4):
    SPEAKERS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SPEAKER'])
    SPEAKERS.pop(speaker_index)
    train_data = IEMOCAP(filtered_speaker=SPEAKERS)
    class_weight = train_data._get_class_weight()
    test_data = IEMOCAP(filtered_speaker=[ARGS.FILTERS_IEMOCAP['SPEAKER'][speaker_index]])

    train_sampler = data.distributed.DistributedSampler(train_data)
    test_sampler = data.distributed.DistributedSampler(test_data)

    train_iter, test_iter = data.DataLoader(train_data, batch_size=batch_size,
                                            sampler=train_sampler), data.DataLoader(test_data, batch_size=batch_size,
                                                                                    sampler=test_sampler)

    return train_iter, test_iter, class_weight


def load_esd_total(batch_size=4, **kwargs):
    """
    load all the esd data (including the train,eval,test)
    :param batch_size:
    :return:
    """
    esd_data = ESDDataset(**kwargs)

    data_iter = data.DataLoader(esd_data, batch_size=batch_size, shuffle=False)
    return data_iter


def load_esd_basic_split(batch_size=4):
    """
    load the ESD dataset for basic training (speaker dependent)
    :param batch_size:
    :return:
    """
    ESD_train = ESDDataset(filtered_split=['train'])
    ESD_eval = ESDDataset(filtered_split=['evaluation'])
    ESD_test = ESDDataset(filtered_split=['test'])

    train_iter, eval_iter, test_iter = data.DataLoader(ESD_train, batch_size=batch_size,
                                                       shuffle=True), data.DataLoader(ESD_eval,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False), data.DataLoader(
        ESD_test, batch_size=batch_size, shuffle=False)
    return train_iter, eval_iter, test_iter


def load_esd_one_speaker_with_eval(eval_speaker_index, test_speaker_index, bathch_size=4):
    """
    load the ESD dataset for Leave-one-speaker-out
    :param eval_speaker_index:
    :param test_speaker_index:
    :param bathch_size:
    :return:
    """
    SPEAKERS = copy.deepcopy(ARGS.FILTERS_ESD['SPEAKER'])
    SPEAKERS.remove(eval_speaker_index)
    SPEAKERS.remove(test_speaker_index)

    train_data = ESDDataset(filtered_speaker=SPEAKERS)
    test_data = ESDDataset(filtered_speaker=test_speaker_index)
    eval_data = ESDDataset(filtered_speaker=eval_speaker_index)

    train_iter, test_iter, eval_iter = data.DataLoader(train_data, batch_size=bathch_size,
                                                       shuffle=True), data.DataLoader(test_data, batch_size=bathch_size,
                                                                                      shuffle=False), data.DataLoader(
        eval_data, batch_size=bathch_size, shuffle=False)

    return train_iter, eval_iter, test_iter


def load_esd_concat_iemocap_one_speaker_with_eval(esd_eval_speaker_index, esd_test_speaker_index, batch_size=4):
    """
    data fusion for esd dataset train
    :param esd_eval_speaker_index: ex. 0001
    :param esd_test_speaker_index: ex. 0002
    :param batch_size:
    :return:
    """

    SPEAKERS = copy.deepcopy(ARGS.FILTERS_ESD['SPEAKER'])
    SPEAKERS.remove(esd_eval_speaker_index)
    SPEAKERS.remove(esd_test_speaker_index)
    iemocap = IEMOCAP(filtered_emotion=[
        "ang", "hap", "sad", "neu", "exc"
    ])

    class_weight = iemocap._get_class_weight()
    esd_train_data = ESDDataset(filtered_emotion=[
        # "Angry", "Happy", "Neutral", "Sad", "Surprise"
        "Angry", "Happy", "Sad", "Neutral",
    ], filtered_speaker=SPEAKERS)

    esd_test_data = ESDDataset(filtered_emotion=[
        # "Angry", "Happy", "Neutral", "Sad", "Surprise"
        "Angry", "Happy", "Sad", "Neutral",
    ], filtered_speaker=[esd_test_speaker_index])

    esd_eval_data = ESDDataset(filtered_emotion=[
        # "Angry", "Happy", "Neutral", "Sad", "Surprise"
        "Angry", "Happy", "Sad", "Neutral",
    ], filtered_speaker=[esd_eval_speaker_index])

    train_data_fusion = data.ConcatDataset([esd_train_data, iemocap])
    print("Num of Samples:", f"IEMOCAP:{len(iemocap)}",
          f"ESD train:{len(esd_train_data)}", f"ESD eval:{len(esd_eval_data)}",
          f"ESD test:{len(esd_test_data)}", f"ESD CONCAT IEMOCAP:{len(train_data_fusion)}", sep='\n', end='\n')
    train_iter, eval_iter, test_iter = data.DataLoader(train_data_fusion, batch_size, shuffle=True), data.DataLoader(
        esd_eval_data, batch_size), data.DataLoader(esd_test_data, batch_size)

    return train_iter, eval_iter, test_iter, class_weight


def load_iemocap_concat_esd_one_session_with_eval(eval_session_index, test_session_index, batch_size=4):
    """
     data fusion for iemocap dataset train(add esd datasets)
    :param eval_session_index:
    :param test_session_index:
    :param batch_size:
    :return:
    """
    SESSIONS = copy.deepcopy(ARGS.FILTERS_IEMOCAP['SESSION'])
    SESSIONS.remove("Session{}".format(eval_session_index))
    SESSIONS.remove("Session{}".format(test_session_index))
    esd = ESDDataset(filtered_emotion=[
        # "Angry", "Happy", "Neutral", "Sad", "Surprise"
        "Angry", "Happy", "Sad", "Neutral",
    ])
    train_data = IEMOCAP(filtered_session=SESSIONS, filtered_emotion=[
        "ang", "hap", "sad", "neu", "exc"
    ])
    class_weight = train_data._get_class_weight()
    eval_data = IEMOCAP(filtered_session=["Session{}".format(eval_session_index)], filtered_emotion=[
        "ang", "hap", "sad", "neu", "exc"
    ])
    test_data = IEMOCAP(filtered_session=["Session{}".format(test_session_index)], filtered_emotion=[
        "ang", "hap", "sad", "neu", "exc"
    ])
    train_data_fusion = data.ConcatDataset([train_data, esd])
    print("Num of Samples:", f"ESD:{len(esd)}",
          f"IEMOCAP train:{len(train_data)}", f"IEMOCAP eval:{len(eval_data)}",
          f"IEMOCAP test:{len(test_data)}", f"IEMOCAP CONCAT ESD:{len(train_data_fusion)}", sep='\n', end='\n')
    train_iter, eval_iter, test_iter = data.DataLoader(train_data_fusion, batch_size=batch_size,
                                                       shuffle=True), data.DataLoader(eval_data,
                                                                                      batch_size=batch_size,
                                                                                      shuffle=False), data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False)

    return train_iter, eval_iter, test_iter, class_weight


if __name__ == '__main__':
    # train_iter, _ = load_IEMOCAP_Dataset_for_TAPT()
    # train_iter, test_iter, class_weights = load_IEMOCAP_Dataset_for_cross_val_leave_one_session_out(session_index=5)
    # train, test = load_RAVDESS_Dataset()
    # for x, length, y in train_iter:
    #     print(x.shape)
    """
    ex: torch.Size([4, 160000]) tensor([ 35200, 160000, 115571,  27560])
    """
    train_iter, test_iter, eval_iter, class_weight = load_esd_concat_iemocap(esd_eval_speaker_index='0001',
                                                                             esd_test_speaker_index='0002',
                                                                             batch_size=10)
