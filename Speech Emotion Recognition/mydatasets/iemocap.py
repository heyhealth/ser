import sys
import os

import numpy as np

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import os
import re
import torch
from torch.utils import data
from collections import Counter
# from mydatasets.feature_extracted_by_Wav2Vec2 import extract_feature_by_Wav2Vec2
# from mydatasets.feature_extracted_by_WavLM import extract_feature_by_WavLM
from mydatasets.feature_acoustic import extract_waveform_from_wav
from mydatasets.text_utils import sentence_to_tokens, build_vocab_from_iterator_wrapper, tokenize_for_ctc
from main.opts import ARGS

"""
    # 'ang': 0, anger 愤怒
    # 'hap': 1, happiness 快乐，幸福
    # 'exc': 2, excitement 激动，兴奋
    # 'sad': 3, sadness 悲伤，悲痛
    # 'fru': 4, frustration 懊恼，沮丧
    # 'fea': 5, fear 害怕，畏惧
    # 'sur': 6, surprise 惊奇，惊讶
    # 'neu': 7, neutral state 中性
    # 'xxx': 8, other 其它
    EMO_LIST = ["ang", "hap", "exc", "sad", "fru", "fea", "sur", "neu", "xxx"]
"""


class IEMOCAP(data.Dataset):

    def __init__(self, data_folder=ARGS.IEMOCAP_DATASET_FOLDER, filtered_session=ARGS.FILTERS_IEMOCAP['SESSION'],
                 filtered_emotion=ARGS.FILTERS_IEMOCAP['EMOTION'], filtered_speaker=ARGS.FILTERS_IEMOCAP['SPEAKER']):
        super(IEMOCAP, self).__init__()
        # convert label to tensor
        self.EMO_LAB_MAP_DIC = dict(zip(
            filtered_emotion, list(range(len(filtered_emotion)))
        ))

        # relabeled excitement samples as happiness.
        # assert "hap" in filtered_emotion
        # self.EMO_LAB_MAP_DIC.update({"exc": self.EMO_LAB_MAP_DIC.get('hap')})

        self.meta_data = []
        # https://huggingface.co/datasets/s3prl/iemocap_split
        # try to implement the above dataset split
        meta_data_keys = ('path', 'label', 'speaker')
        for Session_index in os.listdir(data_folder):
            # filter the needed session
            if Session_index not in filtered_session:
                continue
            label_files_folder_path = os.path.join(data_folder, Session_index, 'dialog', 'EmoEvaluation')
            label_files = [item for item in os.listdir(label_files_folder_path) if item.endswith('.txt')]
            for label_file in label_files:
                label_file_path = os.path.join(label_files_folder_path, label_file)
                wav_names, labels = self._get_path_and_label_from_label_txt(label_file_path)
                for wav_name, label in zip(wav_names, labels):
                    label_ = label
                    # filter the needed model or speaker
                    if label_ not in filtered_emotion:
                        continue
                    path_ = os.path.join(data_folder, Session_index, 'sentences', 'wav',
                                         label_file.strip('.txt'), f'{wav_name}.wav')
                    speaker_ = label_file.split('_')[0]
                    if speaker_ not in filtered_speaker:
                        continue
                    meta_data_values = [path_, label_, speaker_]
                    self.meta_data.append(dict(zip(meta_data_keys, meta_data_values)))

    def _get_path_and_label_from_label_txt(self, label_file_path):
        """
           get wav file name and corresponding label from label.txt file
           :param label_file_path:
           :return:
           """
        wav_names = []
        labels = []
        with open(label_file_path, 'r') as file:
            all_lines = file.readlines()
            for line in all_lines:
                if line.startswith('['):
                    reg = r'\].*\['
                    search_result = re.search(reg, line).group()
                    wav_name_ = search_result.strip('[').strip(']').strip()[:-3].strip()
                    label_ = search_result.strip('[').strip(']').strip()[-3:].strip()
                    wav_names.append(wav_name_)
                    labels.append(label_)
                else:
                    continue

        return wav_names, labels

    def _process_meta_data(self, meta_data):
        feat, length = extract_waveform_from_wav(meta_data['path'], max_sequence_length=ARGS.IEMOCAP_SEQUENCE_LENGTH)
        # feat = extract_feature_by_Wav2Vec2(meta_data['path'])
        # feat = extract_feature_by_WavLM(meta_data['path'])
        label = self.EMO_LAB_MAP_DIC.get(meta_data['label'])
        return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(length, dtype=torch.long), torch.as_tensor(
            label, dtype=torch.long)

    def __getitem__(self, index):
        return self._process_meta_data(self.meta_data[index])

    def __len__(self):
        return len(self.meta_data)

    def _get_class_weight(self):
        # {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 1}
        # ["ang", "hap(exc)", "sad", "neu"]
        all_labels = []
        for meta_data in self.meta_data:
            all_labels.append(self.EMO_LAB_MAP_DIC.get(meta_data['label']))
        counter = Counter(all_labels)
        weights = torch.tensor([counter[index] for index in range(ARGS.NUM_CLASSES_IEMOCAP)]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        return weights


class IEMOCAP_A_T_PAIR(data.Dataset):

    def __init__(self, data_folder=ARGS.IEMOCAP_DATASET_FOLDER, filtered_session=ARGS.FILTERS_IEMOCAP['SESSION'],
                 filtered_emotion=ARGS.FILTERS_IEMOCAP['EMOTION'], filtered_speaker=ARGS.FILTERS_IEMOCAP['SPEAKER'],
                 needed_text=False):
        super(IEMOCAP_A_T_PAIR, self).__init__()
        self.need_text = needed_text
        self.meta_data = []
        self.transcription_tokens_list = []
        self.padding_token_values = -100
        self.token_length = 300
        meta_data_keys = ('path', 'label', 'speaker', 'transcription')
        for session_index in os.listdir(data_folder):
            if session_index not in filtered_session:
                continue
            label_files_folder_path = os.path.join(data_folder, session_index, 'dialog', 'EmoEvaluation')
            label_files = [item for item in os.listdir(label_files_folder_path) if item.endswith('.txt')]
            sentences_wav_path = os.path.join(data_folder, session_index, 'sentences', 'wav')
            transcription_path = os.path.join(data_folder, session_index, 'dialog', 'transcriptions')

            for label_file in label_files:
                label_file_path = os.path.join(label_files_folder_path, label_file)
                wav_names, labels = self._get_path_and_label_from_label_txt(label_file_path)
                transcription_file_path = os.path.join(transcription_path, label_file)
                for wav_name, label in zip(wav_names, labels):
                    if label not in filtered_emotion:
                        continue
                    label_ = label
                    # the wav file's path
                    path_ = os.path.join(sentences_wav_path, label_file.strip('.txt'), f'{wav_name}.wav')
                    speaker_ = label_file.split('_')[0]
                    if speaker_ not in filtered_speaker:
                        continue
                    transcription_ = self._get_transcription_by_wav_name(transcription_file_path, wav_name)
                    self.transcription_tokens_list.append(sentence_to_tokens(transcription_))
                    meta_data_values = [path_, label_, speaker_, transcription_]
                    self.meta_data.append(dict(zip(meta_data_keys, meta_data_values)))

        self.vocab = build_vocab_from_iterator_wrapper(self.transcription_tokens_list)

    def __getitem__(self, index):
        # return self._process_meta_data_for_test(self.meta_data[index])
        return self._process_meta_data(self.meta_data[index])

    def __len__(self):
        return len(self.meta_data)

    def _process_meta_data(self, meta_data):
        feat, length = extract_waveform_from_wav(meta_data['path'], max_sequence_length=ARGS.IEMOCAP_SEQUENCE_LENGTH)
        # feat = extract_feature_by_Wav2Vec2(meta_data['path'])
        # feat = extract_feature_by_WavLM(meta_data['path'])
        label = self.EMO_LAB_MAP_DIC.get(meta_data['label'])
        # need process the transcript
        # tokenize by the word is WRONG
        # transcript_vocab_idxs = self.vocab.lookup_indices(sentence_to_tokens(meta_data['transcription']))
        # tokenize by the character
        transcript_tokenize = tokenize_for_ctc(meta_data['transcription'])
        # padding the tokens
        if len(transcript_tokenize.input_ids) < self.token_length:
            transcript_tokenize.input_ids = np.pad(transcript_tokenize.input_ids,
                                                   (0, self.token_length - len(transcript_tokenize.input_ids)),
                                                   mode='constant',
                                                   constant_values=(0, self.padding_token_values))
            transcript_tokenize.attention_mask = np.pad(transcript_tokenize.attention_mask,
                                                        (
                                                            0, self.token_length - len(
                                                                transcript_tokenize.attention_mask)),
                                                        mode='constant',
                                                        constant_values=(0, self.padding_token_values))

        else:
            transcript_tokenize.input_ids = transcript_tokenize.input_ids[0:self.token_length]
            transcript_tokenize.attention_mask = transcript_tokenize.attention_mask[0:self.token_length]

        if self.need_text:
            return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(length,
                                                                               dtype=torch.long), torch.as_tensor(
                label, dtype=torch.long), torch.as_tensor(transcript_tokenize.input_ids, dtype=torch.long)

        return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(length, dtype=torch.long), torch.as_tensor(
            label, dtype=torch.long)

    def _get_transcription_by_wav_name(self, trans_file_path, wav_name):
        """
        get the corresponding transcript by the wav name
        :param trans_file_path:
        :param wav_name:
        :return: corresponding transcript
        """
        with open(trans_file_path, 'r') as file:
            all_lines = file.readlines()
            for line in all_lines:
                if line.startswith(wav_name):
                    transcript = line.split(']:')[-1]
                    return transcript
        return None

    def _get_path_and_label_from_label_txt(self, label_file_path):
        """
           get wav file name and corresponding label from label.txt file
           :param label_file_path:
           :return:
           """
        wav_names = []
        labels = []
        with open(label_file_path, 'r') as file:
            all_lines = file.readlines()
            for line in all_lines:
                if line.startswith('['):
                    reg = r'\].*\['
                    search_result = re.search(reg, line).group()
                    wav_name_ = search_result.strip('[').strip(']').strip()[:-3].strip()
                    label_ = search_result.strip('[').strip(']').strip()[-3:].strip()
                    wav_names.append(wav_name_)
                    labels.append(label_)
                else:
                    continue

        return wav_names, labels

    def _get_vocab_size(self):
        return len(self.vocab)

    def _get_class_weight(self):
        # {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 1}
        # ["ang", "hap(exc)", "sad", "neu"]
        all_labels = []
        for meta_data in self.meta_data:
            all_labels.append(self.EMO_LAB_MAP_DIC.get(meta_data['label']))
        counter = Counter(all_labels)
        weights = torch.tensor([counter[index] for index in range(ARGS.NUM_CLASSES_IEMOCAP)]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        return weights


def get_vocal_size_IEMOCAP():
    iemocap_datasets = IEMOCAP_A_T_PAIR()
    return iemocap_datasets._get_vocab_size()


def test_the_IEMOCAP():
    iemocap = IEMOCAP_A_T_PAIR(needed_text=True)
    token_length = []
    for meta_data in iemocap:
        token_length.append(len(meta_data[3]))
        print(meta_data[3])
    token_length_tensor = torch.tensor(token_length, dtype=torch.float32)
    print(token_length_tensor.max())
    print(token_length_tensor.mean())
    print(token_length_tensor.min())
    # tensor(119.)
    # tensor(15.4430)
    # tensor(1.)


if __name__ == '__main__':
    # print(get_vocal_size_IEMOCAP())

    test_the_IEMOCAP()
