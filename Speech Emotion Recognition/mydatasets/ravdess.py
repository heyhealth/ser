import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import os
import torch
from torch.utils import data
# from datasets.feature_extracted_by_Wav2Vec2 import extract_feature_by_Wav2Vec2
# from datasets.feature_extracted_by_WavLM import extract_feature_by_WavLM
from mydatasets.feature_acoustic import extract_waveform_from_wav
from main.opts import ARGS
from collections import Counter

"""
Audio-only (16bit, 48kHz .wav)

    Speech file (Audio_Speech_Actors_01-24.zip, 215 MB) contains 1440 files: 60 trials per actor x 24 actors = 1440. 
    Song file (Audio_Song_Actors_01-24.zip, 198 MB) contains 1012 files: 44 trials per actor x 23 actors = 1012.

Filename identifiers 

    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).


"""

RAVDESS_DATASET_Modality_FOLDER = ARGS.RAVDESS_DATASET_Modality_FOLDER
FILTERS = ARGS.FILTERS_RAVDESS
#
EMO_LAB_MAP_DIC = dict(zip(
    FILTERS['Emotion'], list(range(len(FILTERS['Emotion'])))
))
# merge
assert '02' in EMO_LAB_MAP_DIC.keys()
EMO_LAB_MAP_DIC.update({'02': EMO_LAB_MAP_DIC.get('01')})


class RAVDESS(data.Dataset):

    def __init__(self, Modality_folder=RAVDESS_DATASET_Modality_FOLDER, filters=FILTERS):
        """
        :param Modality_folder:
        :param filter: {Modality:[],Vocal_channel[],Emotion:[],Emotional_intensity:[]...}
        """
        super(RAVDESS, self).__init__()
        self.meta_data = []
        for Vocal_channel in os.listdir(Modality_folder):
            for Actor_index in os.listdir(os.path.join(Modality_folder, Vocal_channel)):
                for wav_name in os.listdir(os.path.join(Modality_folder, Vocal_channel, Actor_index)):
                    wav_path = os.path.join(Modality_folder, Vocal_channel, Actor_index, wav_name)
                    meta_data_values = wav_name.strip('.wav').split('-')
                    if not self._filter_fun(meta_data_values, filters):
                        continue
                    meta_data_values.append(wav_path)
                    self.meta_data.append(dict(zip(filters.keys(), meta_data_values)))

    def __getitem__(self, index):
        return self._process_meta_data(self.meta_data[index])

    def __len__(self):
        return len(self.meta_data)

    def _filter_fun(self, values, filters):
        assert len(values) + 1 == len(filters)
        FLAG = True
        for value_, filter_ in zip(values, filters.keys()):
            if value_ not in filters[filter_]:
                FLAG = False
                break
        return FLAG

    def _process_meta_data(self, meta_data):

        feat, length = extract_waveform_from_wav(meta_data['wav_path'],
                                                 max_sequence_length=ARGS.RAVDESS_SEQUENCE_LENGTH)
        # feat = extract_feature_by_Wav2Vec2(meta_data['wav_path'], resampled=True)
        # feat = extract_feature_by_WavLM(meta_data['wav_path'], resampled=True)
        label = EMO_LAB_MAP_DIC.get(meta_data['Emotion'])
        return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(length, dtype=torch.long), torch.as_tensor(
            label, dtype=torch.long)

    def _get_class_weight(self):
        # {'01': 0, '03': 1, '04': 2, '05': 3, '06': 4, '07': 5, '08': 6, '02': 0}
        all_labels = []
        for meta_data in self.meta_data:
            all_labels.append(EMO_LAB_MAP_DIC.get(meta_data['Emotion']))
        counter = Counter(all_labels)
        weights = torch.tensor([counter[index] for index in range(ARGS.NUM_CLASSES_RAVDESS)]).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        return weights


def test_the_RAVDESS():
    ravdess = RAVDESS()
    for meta_data, label in ravdess:
        print(meta_data.shape, end='\n')


if __name__ == '__main__':
    test_the_RAVDESS()
