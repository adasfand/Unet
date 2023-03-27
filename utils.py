import numpy as np

def preprocess_mask(mask):
    # print(type(mask))
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask
"""
数据集包含像素级三图分割。 对于每个图像，都有一个带掩码的关联PNG文件。 
掩码的大小等于相关图像的大小。 
遮罩图像中的每个像素可以采用以下三个值之一：
1、2或3.1表示图像的该像素属于``宠物''类别，``2''属于背景类别，``3''属于边界类别。
"""