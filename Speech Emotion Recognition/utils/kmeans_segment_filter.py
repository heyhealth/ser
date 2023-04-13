import librosa
from sklearn.cluster import KMeans
import numpy as np
import torch
from scipy.spatial.distance import euclidean


# TODO: use the k-means clustering methods to filte efficient speech segment
# have negative effect on SER

def waveform_split_by_ms(waveform, sample_rate, segment_length_ms=500, overlap=125):
    """
    split the waveform by the fix time (default:500ms) and overlap is 25%(default:125ms)
    :param waveform:
    :param sample_rate:
    :param segment_length_ms:
    :param overlap:
    :return: [num_segment,segment_length] ex.[8,8000]
    """
    segment_size = int((segment_length_ms / 1000) * sample_rate)
    segment_shift = int((overlap / 1000) * sample_rate)

    segments_list = []

    for i in range(len(waveform) // segment_shift):
        if i * segment_shift + segment_size >= len(waveform):
            break
        segments_list.append(waveform[i * segment_shift:i * segment_shift + segment_size].unsqueeze(0))
    return segments_list


def get_the_K_nearest_item_dist_cluster_center(X, K):
    kmeans = KMeans(init='k-means++', n_clusters=K, n_init=10, random_state=3407).fit(X)
    k_nearest_x_dist_center_list = []
    for center in kmeans.cluster_centers_:
        item = X[0]
        min_dist = np.Inf
        for x in X:
            if euclidean(center, x) < min_dist:
                item = x
                min_dist = euclidean(center, x)
        k_nearest_x_dist_center_list.append(item)

    return k_nearest_x_dist_center_list, kmeans.labels_


def recombine_waveform(waveform, sr, K=3, segment_length_ms=500, overlap=125):
    _, to_cluster = waveform_split_by_ms(waveform, sr, segment_length_ms=segment_length_ms, overlap=overlap)
    k_key_segment, _ = get_the_K_nearest_item_dist_cluster_center(to_cluster, K)
    return np.concatenate(k_key_segment)



def k_key_spectrogram_kmeans(waveform,sample_rate,segment_length_ms,overlap,K=3):


    segments = waveform_split_by_ms(waveform,sample_rate,segment_length_ms,overlap)
    segments_mfcc = []
    for segment in segments:

        mfcc = librosa.feature.mfcc(segment)
        segments_mfcc.append(mfcc)

    get_the_K_nearest_item_dist_cluster_center(segments_mfcc,K=K)



if __name__ == '__main__':
    wav_path = r"E:\Datasets\IEMOCAP\raw_unzip\Session5\sentences\wav\Ses05F_impro04\Ses05F_impro04_F040.wav"

    waveform, sr = librosa.load(wav_path)

    print(
        sr
    )
    recombined_waveform = recombine_waveform(torch.tensor(waveform), sr, K=3)

    print(
        recombined_waveform.shape
    )
