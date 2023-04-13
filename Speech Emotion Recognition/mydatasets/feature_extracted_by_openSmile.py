import os
import numpy as np

CACHE_PATH = r"E:\Datasets\RAVDESS\.cache"
CONFIG_PATH = r"D:\Tools\openSMILE\opensmile-3.0-win-x64\config\emobase\emobase.conf"


def extract_feature_by_openSmile(source_wav_path, config_path=CONFIG_PATH, cache_path=CACHE_PATH):
    """
    extract the feature from given speech by the openSmile Toolkits
    :param source_wav_path:
    :param config_path: the feature to choice
    :param cache_path: cache the files in processing
    :return: the feature , np.array
    """
    # rewrite_wav_by_ffmpeg
    wav_name_ = source_wav_path.split('\\')[-1]
    temp_wav_path = os.path.join(cache_path, wav_name_)
    rewrite_command = r"ffmpeg -i {} -ar 16000 {}".format(source_wav_path, temp_wav_path)
    os.system(rewrite_command)
    print("Rewrite The Audio File Done!")
    temp_csv_path = os.path.join(cache_path, wav_name_.replace('wav', 'csv'))
    # extract the feature to csv
    extract_command = r"cd D:\Tools\openSMILE\opensmile-3.0-win-x64\bin && SMILExtract -C {} -I {} -O {}".format(
        config_path, temp_wav_path, temp_csv_path)
    os.system(extract_command)
    print("Extract The Audio Feature By openSmile Done!")
    # load the feature from csv
    with open(temp_csv_path, 'r', encoding='utf-8') as file:
        last_line = file.readlines()[-1]
        filter_ = last_line.split(',')[1:-2]
        feature = [float(item) for item in filter_]
    return np.array(feature)


def do_test_extract_feature_by_openSmile():
    wav_path = r"E:\Datasets\RAVDESS\raw\Audio-only-files\Audio_Song_Actors_01-24\Actor_01\03-02-01-01-02-01-01.wav"
    feat = extract_feature_by_openSmile(wav_path, CONFIG_PATH, CACHE_PATH)
    print(feat.shape)
