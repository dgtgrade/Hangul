import random
import time
import numpy as np
import tensorflow as tf

#
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt


#
FONT_SIZE = 72
FONT_WIDTH = 72
FONT_HEIGHT = 72
FONT_LENGTH = FONT_WIDTH * FONT_HEIGHT

#
한글초성개수 = 19
한글중성개수 = 21
한글종성개수 = 28
한글총개수 = 한글초성개수 * 한글중성개수 * 한글종성개수

#
유코한글시작 = 0xac00


def 한글자_분리번(한글자):
    자번 = ord(한글자) - 유코한글시작
    return 자번_분리번(자번)


def 자번_분리번(자번):    
    종성번 = 자번 % 한글종성개수
    중성번 = (자번 // 한글종성개수) % 한글중성개수
    초성번 = (자번 // 한글종성개수) // 한글중성개수

    return [초성번, 중성번, 종성번]


def 초중종번_조합번(초성번, 중성번, 종성번):
    유코번 = 유코한글시작 + \
          초성번 * 한글중성개수 * 한글종성개수 + \
          중성번 * 한글종성개수 + \
          종성번

    return 유코번


def 랜덤자():
    초성번 = int(random.random() * 한글초성개수)
    중성번 = int(random.random() * 한글중성개수)
    종성번 = int(random.random() * 한글종성개수)

    return chr(초중종번_조합번(초성번, 중성번, 종성번))


def load_한글BMPs():

    #
    font = ImageFont.truetype("fonts/NanumGothicCoding-2.0.ttf", FONT_SIZE)

    #
    # 72*72*10000*4B <= 300MB
    #
    한글BMPs = np.zeros((한글총개수, FONT_WIDTH, FONT_HEIGHT))

    for 초번 in range(한글초성개수):
        for 중번 in range(한글중성개수):
            for 종번 in range(한글종성개수):
                #
                유코번 = 초중종번_조합번(초번, 중번, 종번)
                자번 = 유코번 - 유코한글시작
                자 = chr(유코번)

                img = Image.new("RGB", (FONT_WIDTH, FONT_HEIGHT))
                draw = ImageDraw.Draw(img)
                draw.text((0, 0), 자, font=font)
                #
                w, h = draw.textsize(자, font=font)
                assert w <= FONT_WIDTH and h <= FONT_HEIGHT
                #
                imgnd = np.mean(np.array(img), axis=2)  # rgb to gray
                한글BMPs[자번, :, :] = imgnd[:, :]
                #
                # plt.imshow(한글BMPs[자번, :, :], cmap='gray', vmin=0, vmax=1, interpolation='none')
                # plt.draw()
                # plt.show(block=False)
                # plt.waitforbuttonpress()

    return 한글BMPs

my_한글_BMPs = load_한글BMPs()
inputs = my_한글_BMPs.reshape(한글총개수, -1)
codes = np.arange(한글총개수)
codes_초중종 = np.array([자번_분리번(자번) for 자번 in range(한글총개수)])
labels_초 = np.zeros((한글총개수, 한글초성개수))
labels_중 = np.zeros((한글총개수, 한글중성개수))
labels_종 = np.zeros((한글총개수, 한글종성개수))
labels_초[np.arange(한글총개수), codes_초중종[:, 0]] = 1
labels_중[np.arange(한글총개수), codes_초중종[:, 1]] = 1
labels_종[np.arange(한글총개수), codes_초중종[:, 2]] = 1
# print(labels_초)
# print(labels_중)
# print(labels_종)


def fc_layer(prev_layer, prev_n_nodes, n_nodes, activate=tf.nn.tanh, name=""):

    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([prev_n_nodes, n_nodes], stddev=0.1), name="weights")
        b = tf.Variable(tf.zeros([n_nodes]), name="biases")

        z = tf.add(tf.matmul(prev_layer, w), b, name='zs')
        a = activate(z, name='as')

    return a


#
input_layer = tf.placeholder(tf.float32, [None, FONT_LENGTH], name='input')
#
hidden_layer_0 = fc_layer(input_layer, FONT_LENGTH, 256, name='hidden_0')
hidden_layer_1 = fc_layer(hidden_layer_0, 256, 256, name='hidden_1')
hidden_layer_2 = fc_layer(hidden_layer_1, 256, 256, name='hidden_2')
hidden_layer_3 = fc_layer(hidden_layer_2, 256, 256, name='hidden_3')
hidden_layer_4 = fc_layer(hidden_layer_3, 256, 256, name='hidden_4')
#
output_layer_초 = fc_layer(hidden_layer_4, 256, 한글초성개수, name='output_i')
output_layer_중 = fc_layer(hidden_layer_4, 256, 한글중성개수, name='output_m')
output_layer_종 = fc_layer(hidden_layer_4, 256, 한글종성개수, name='output_f')
#
output_layer = tf.cast(tf.argmax(output_layer_초, 1) * 한글중성개수 * 한글종성개수 + \
                       tf.argmax(output_layer_중, 1) * 한글종성개수 + \
                       tf.argmax(output_layer_종, 1), tf.int32, name='output_layer')
#
output_label_초 = tf.placeholder(tf.float32, [None, 한글초성개수], name='output_i')
output_label_중 = tf.placeholder(tf.float32, [None, 한글중성개수], name='output_m')
output_label_종 = tf.placeholder(tf.float32, [None, 한글종성개수], name='output_m')
#
output_code = tf.placeholder(tf.int32, [None], name='output_code')
#
loss_초 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_초, output_label_초, name='loss_i'))
loss_중 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_중, output_label_중, name='loss_m'))
loss_종 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_종, output_label_종, name='loss_f'))
loss_총 = loss_초 + loss_중 + loss_종
#
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_총)
#
correct_prediction = tf.equal(output_code, output_layer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

#
sess.run(tf.initialize_all_variables())


#
VALID_RATIO = 0.05
valid_sets = np.random.random(한글총개수) < VALID_RATIO
train_sets = -valid_sets

for i in range(100000):
    result = sess.run([accuracy, loss_총, loss_초, loss_중, loss_종, optimizer],
                      feed_dict={
                          input_layer: inputs[train_sets],
                          output_label_초: labels_초[train_sets],
                          output_label_중: labels_중[train_sets],
                          output_label_종: labels_종[train_sets],
                          output_code: codes[train_sets],
                      })
    if i % 10 == 0:
        valid_acc = sess.run([accuracy],
                             feed_dict={
                                 input_layer: inputs[valid_sets],
                                 output_label_초: labels_초[valid_sets],
                                 output_label_중: labels_중[valid_sets],
                                 output_label_종: labels_종[valid_sets],
                                 output_code: codes[valid_sets]
                             })
        print(i, valid_acc, result[:-1])

