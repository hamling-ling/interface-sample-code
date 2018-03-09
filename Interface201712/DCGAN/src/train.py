# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import os

import matplotlib
import numpy as np
import tensorflow as tf
from dcgan import DCGAN

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import glob
from util import preprocess_img

SIZE = 64

IMG_DIR_TRAIN="training"
IMG_DIR_EVAL="evaluation"

def load_images(img_type, is_distortion):
    """
    画像を読み込む

    :return:
    """
    # (1)画像のロード
    if is_distortion:
        d = "_d"
    else:
        d = ""
    filepaths = glob.glob("../data/input/{0}/*[0-9]{1}.npy".format(img_type, d))
    filepaths.sort()

    images = [preprocess_img(filepath=filepath, size=SIZE) for filepath in filepaths];
    print("{0}枚の画像を取り込みました。".format(len(images)))
    return np.array(images, dtype=np.float32)

def save_images(source_images, generated_images, filename):
    plt.clf()
    for i in range(len(generated_images)):
        plt.subplot(8,8,i+1)
        plt.plot(source_images[i].reshape(-1))
        plt.plot(generated_images[i].reshape(-1))
        plt.savefig(filename)
    plt.clf()

def train():
    """
    学習データを構築する。
    """
    # 画像をデータセットから読み込む
    imgs = load_images(IMG_DIR_TRAIN, is_distortion=True)
    noises = load_images(IMG_DIR_TRAIN, is_distortion=False)

    with tf.Session() as sess:
        # (3)DCGANネットワークの生成
        batch_size = 64
        dcgan = DCGAN(
            generator_layers=[1024, 512, 256, 128],
            discriminator_layer=[64, 128, 256, 512],
            batch_size=batch_size,
            image_inputs=tf.placeholder(tf.float32, [batch_size, 1, SIZE, 1]),
            noise_inputs=tf.placeholder(tf.float32, [batch_size, 1, SIZE, 1])
        )
        sess.run(tf.global_variables_initializer())

        # (4)ファイル保存の準備
        g_saver = tf.train.Saver(dcgan.generator.variables)
        d_saver = tf.train.Saver(dcgan.discriminator.variables)

        maxstep = 10000
        N = len(imgs)

        # (5)サンプル出力の準備
        sample_z = load_images(IMG_DIR_EVAL, is_distortion=False)[0:64]
        images = dcgan.sample_images(8, 8, inputs=sample_z)

        os.makedirs('../data/generated_images/', exist_ok=True)

        save_images(noises[:64], imgs[:64], "../data/generated_images/teacher.png")

        # (6)学習
        for step in range(maxstep):
            permutation = np.random.permutation(N)
            imgs_batch = imgs[permutation[0:batch_size]]
            noises_batch = noises[permutation[0:batch_size]]
            g_loss, d_loss = dcgan.fit_step(sess=sess, image_inputs=imgs_batch, noise_inputs=noises_batch)

            # 100 stepごとに学習結果を出力する。
            if step % 100 == 0:
                filename = os.path.join('../data/', "generated_images", '%05d.png' % step)
                images2save = sess.run(images)
                save_images(sample_z, images2save, filename)
                print("Generator loss: {} , Discriminator loss: {}".format(g_loss, d_loss))

        # (7)学習済みモデルのファイル保存
        os.makedirs('../data/models/', exist_ok=True)
        g_saver.save(sess=sess, save_path="../data/models/g_saver.ckpg")
        d_saver.save(sess=sess, save_path="../data/models/d_saver.ckpg")


if __name__ == '__main__':
    train()
