# IEMOCAP
"""
all classes 10034=1817+1810+2135+2102+2170
4 classes ["ang", "hap", "sad", "neu"] 4490=942(135)+813(117)+1000(135)+793(65)+942(143) (hap in bracket)
4 classes ["ang", "hap", "sad", "neu"] 4490=(1103,595,1084,1708)
4 classes(merge exc(all 1041)->hap) ["ang", "hap(exc)", "sad", "neu"] 5531=1085(229,278,194,382)+1023(137,327,197,362)+1151(240,286,305,320)+1031(327,303,143,258)+1241(170,442,245,384)
4 classes(merge exc(all 1041)->hap) ["ang", "hap(exc)", "sad", "neu"] 5531=(1103,1636,1084,1708)
"""

# RAVDESS
"""


"""

# ESD

"""
Total samples(Chinese):
    17500
    10个人，每个人1750个样本；
Total classes:
    {'Angry': 0, 'Happy': 1, 'Neutral': 2, 'Sad': 3, 'Surprise': 4}
文件结构：
    人（10）->情绪（5）->划分（train:300;evaluation:20;test:30）->音频文件
"""

# Concat IEMOCAP-ESD
"""
len-iemocap:5531
len-train-esd:11200
len-test-esd:1400
len-eval-esd:1400
len-fusion:16731s
"""
