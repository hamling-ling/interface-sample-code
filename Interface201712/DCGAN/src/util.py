# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np


def preprocess_img(filepath, size):
    """
    画像の前処理。読み込んでsizeの大きさに縦横をリサイズする

    :param img:
    :return:
    """
    # (2)画像の前処理
    # 画像の縦横を引数のsizeの大きさにリサイズし、輝度を0～1の範囲から-1～1に変更
    return np.load(filepath)
