import wave
import os
import librosa


def is_wav(f):
    res = True
    try:
        wave.open(f)
    except wave.Error as e:
        res = False
    return res


def pcm2wav(pcm_file, save_file, channels=1, bits=16, sample_rate=16000):
    """ pcm转换为wav格式

        Args:
            pcm_file pcm文件
            save_file 保存文件
            channels 通道数
            bits 量化位数，即每个采样点占用的比特数
            sample_rate 采样频率
    """
    if is_wav(pcm_file):
        raise ValueError('"' + str(pcm_file) + '"' +
                         " is a wav file, not pcm file! ")

    pcmf = open(pcm_file, 'rb')
    pcmdata = pcmf.read()
    pcmf.close()

    if bits % 8 != 0:
        raise ValueError("bits % 8 must == 0. now bits:" + str(bits))

    wavfile = wave.open(save_file, 'wb')

    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits // 8)
    wavfile.setframerate(sample_rate)

    wavfile.writeframes(pcmdata)
    wavfile.close()


def convert_dir(root, ext=".pcm", **kwargs):
    """ 把一个文件夹内的pcm，统统加上头

        Args:
            root 文件夹根目录
            ext pcm文件的扩展名
    """

    from tqdm import tqdm

    src_files = [os.path.join(dir_path, f)
                 for dir_path, _, files in os.walk(root)
                 for f in files
                 if os.path.splitext(f)[1] == ext]

    for src_file in tqdm(src_files, ascii=True):
        try:
            wav_file = os.path.splitext(src_file)[0] + ".wav"
            pcm2wav(src_file, wav_file, **kwargs)
        except Exception as e:
            print('Convert fail: ' + src_file)
            print(e)


if __name__ == '__main__':
    # pcm = r'1.pcm'
    # wav = pcm[:-4] + '.wav'
    # pcm2wav(pcm, wav)
    # convert_dir(r'/path/to/pcm/dir', '.pcm')

    pcm = r"/home/user2/health/tmp/0004_000752.pcm"
    original_wav_path = r"/home/user2/health/tmp/0004_000752.wav"
    tran_save_path = r"/home/user2/health/tmp/0004_000752trans.wav"
    pcm2wav(pcm, tran_save_path)
    y1, sr1 = librosa.load(original_wav_path, sr=None)
    y2, sr2 = librosa.load(tran_save_path, sr=None)
    print(y1.shape, sr1)
    print(y2.shape, sr2)
    # (64475,)
    # print("hello")
