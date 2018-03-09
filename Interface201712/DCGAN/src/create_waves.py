import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

wave_num=10*1000
sz=1024
def createGraph():
    rnd_f=0.04+0.16*tf.random_uniform([wave_num,1])
    rnd_p=np.pi*tf.random_uniform([wave_num,1])

    idx = tf.reshape(tf.constant(np.arange(sz,dtype=np.float32)),[1,sz])
    y=idx*rnd_f*np.pi+rnd_p
    return tf.sin(y)

def outputWave(path):
    with tf.Session() as sess:
        g=createGraph()
        waves=sess.run(g)

        for i in range(len(waves)):
            fn = "{0}/{1:0>4}.npy".format(path, i)
            np.save(fn, waves[i])

            fn_d = "{0}/{1:0>8}_d.npy".format(path,i)
            np.save(fn_d, distortion(waves[i]))

def distortion(x):
    return np.sign(x)*(1-np.exp(-np.abs(x*8)))

out_train="../data/input/training"
out_validation = "../data/input/validation"
out_evaluation = "../data/input/evaluation"

os.makedirs("../data/input", exist_ok=True)
os.makedirs(out_train, exist_ok=True)
os.makedirs(out_validation, exist_ok=True)
os.makedirs(out_evaluation, exist_ok=True)

outputWave(out_train)
outputWave(out_validation)
outputWave(out_evaluation)
