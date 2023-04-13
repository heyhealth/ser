import sys
import os

sys.path.append(os.path.dirname((os.path.dirname(os.path.realpath(__file__)))))
import torch
from torch.utils.data import Dataset
from main.opts import ARGS
from mydatasets.feature_acoustic import extract_waveform_from_wav

"""
{'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
"""


class ESDDataset(Dataset):

    def __init__(self, data_folder=ARGS.ESD_DATASET_FOLDER, filtered_split=ARGS.FILTERS_ESD['SPLIT'],
                 filtered_emotion=ARGS.FILTERS_ESD['EMOTION'],
                 filtered_speaker=ARGS.FILTERS_ESD['SPEAKER']):
        super(ESDDataset, self).__init__()

        self.EMO_LAB_MAP_DIC = dict(zip(
            filtered_emotion, list(range(len(filtered_emotion)))
        ))

        self.meta_data = []
        self.meta_keys = ('path', 'label', 'speaker')
        for speaker_index in os.listdir(data_folder):
            if speaker_index not in filtered_speaker or speaker_index.endswith('.txt'):
                continue
            for emotion in os.listdir(os.path.join(data_folder, speaker_index)):

                if emotion not in filtered_emotion or emotion.endswith('.txt'):
                    continue
                for split in os.listdir(os.path.join(data_folder, speaker_index, emotion)):
                    if split not in filtered_split:
                        continue
                    cur_dir = os.path.join(data_folder, speaker_index, emotion, split)
                    for wav_file in os.listdir(cur_dir):
                        file_path = os.path.join(cur_dir, wav_file)
                        speaker = speaker_index
                        meta_values = [file_path, emotion, speaker]
                        self.meta_data.append(dict(zip(self.meta_keys, meta_values)))

    def _process_meta_data(self, meta_data):

        feat, length = extract_waveform_from_wav(meta_data['path'], max_sequence_length=ARGS.ESD_SEQUENCE_LENGTH)
        # feat = extract_feature_by_Wav2Vec2(meta_data['path'])
        # feat = extract_feature_by_WavLM(meta_data['path'])
        label = self.EMO_LAB_MAP_DIC.get(meta_data['label'])
        return torch.as_tensor(feat, dtype=torch.float32), torch.as_tensor(length, dtype=torch.long), torch.as_tensor(
            label, dtype=torch.long)

    def __getitem__(self, index):
        return self._process_meta_data(self.meta_data[index])

    def __len__(self):
        return len(self.meta_data)


def test_ESD():
    esd = ESDDataset()
    # for meta_data in esd:
    #     print(meta_data[0].shape)
    # for meta_data, label, speaker in esd:
    #     print(meta_data, label, end='\n')
    print(len(esd))

if __name__ == '__main__':
    test_ESD()
