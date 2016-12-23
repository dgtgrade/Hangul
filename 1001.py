# -*- coding: utf-8 -*-

import sys
import random
import math
# import time
import numpy as np
import tensorflow as tf

#
from PIL import Image, ImageFont, ImageDraw
#
from fontTools.ttLib import TTFont

import matplotlib.pyplot as plt
#
ONLY_ONE_EXAMPLE_PER_MINI_BATCH = False
#
MINI_BATCH_SIZE = 1000
SHOW_STATUS_PER_EPOCH = 1
LEARNING_RATE = 1 * 1e-4
N_TRAIN_FONTS = None  # None for All
BATCH_NORMALIZATION_DECAY = 0.95
TOTAL_EPOCHS = 10000
SHOW_FONT = False
N_DISTORTIONS = 4

SAVE_MODEL_PER_EPOCHS = 10
MODEL_DIR = "model"
LOAD_MODEL = True

#
# 글꼴들
FONT_DIR = "fonts"
train_fonts = [
    # 네이버, 나눔 글꼴
    # http://hangeul.naver.com/2016/nanum
    "NanumGothicCoding-2.0.ttf",  # 가장 인식하기 쉬운 폰트?
    "NanumBarunGothic.ttf",
    "NanumBarunpenB.ttf",
    "NanumBrush.ttf",
    "NanumMyeongjo.ttf",
    "NanumPen.ttf",
    "NanumSquareB.ttf",
    # 우아한 형제들, 배달의 민족 글꼴
    # http://font.woowahan.com/
    "BMDOHYEON_ttf.ttf",
    "BMHANNA_11yrs_ttf.ttf",
    "BMJUA_ttf.ttf",
    "BMYEONSUNG_ttf.ttf",
    # 윤태호, 미생체
    # https://storyfunding.daum.net/project/334
    "Daum_Regular.ttf",
    # http://cast.yanolja.com/detail/2171
    "Yanolja_Regular.ttf",
    # http://www.bing.co.kr/pr/bing_font.aspx
    "Binggrae.ttf",
    # http://www.asan.go.kr/font
    "YiSunShin_Regular.ttf",
    # http://goyang.go.kr/kr/intro/sub03/09/
    "Goyang.ttf",
    # http://www.dxkorea.co.kr/shop/main/index.php
    "DXSeNB-KSCpc-EUC-H.ttf"
]

valid_fonts = [
    # http://specialsister.tistory.com/9
    "HGSS.ttf",
    # http://www.hoondesign.co.kr/sub/hoonfont.php
    "HoonWhitecatR.ttf",
]

#
FONT_SIZE = 32
FONT_WIDTH = 40
FONT_HEIGHT = 40
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

    자번s = np.atleast_1d(자번)
    분리번s = np.empty((len(자번s), 3), dtype=np.int)
    for i, 자번 in enumerate(자번s):
        종성번 = 자번 % 한글종성개수
        중성번 = (자번 // 한글종성개수) % 한글중성개수
        초성번 = (자번 // 한글종성개수) // 한글중성개수
        분리번s[i, :] = [초성번, 중성번, 종성번]

    if len(자번s) == 1:
        return 분리번s[0, 0], 분리번s[0, 1], 분리번s[0, 2]
    else:
        return 분리번s[:, 0], 분리번s[:, 1], 분리번s[:, 2]


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


def show_font(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none')
    plt.draw()
    plt.show(block=False)
    plt.waitforbuttonpress()


def load_한글_BMPs(fonts, n_distorts):

    # TODO: list를 이런식으로 써도 성능에 문제 없는지?
    한글_BMPs = []
    한글_codes = []
    체크번s = [0, 100, 500, 1000, 7000, 9000]

    #
    for font_i, font_fp in enumerate(fonts):

        HQ_RATIO = 2
        font = ImageFont.truetype(FONT_DIR + "/" + font_fp, FONT_SIZE * HQ_RATIO)
        ft_font = TTFont(FONT_DIR + "/" + font_fp)
        # http://www.programcreek.com/python/example/81172/fontTools.ttLib
        # TODO: getcmap이 뭐하는 것인지 이해하고 정말 아래 코드 그대로 쓰면 되는지 체크하기
        unicode_cmap = ft_font['cmap'].getcmap(3, 10)
        if not unicode_cmap:
            unicode_cmap = ft_font['cmap'].getcmap(3, 1)

        n_chrs = 0

        #
        # 72*72*10000*4B <= 300MB
        #
        print("loading... {}".format(font_fp))
        for 초번 in range(한글초성개수):
            for 중번 in range(한글중성개수):
                for 종번 in range(한글종성개수):
                    #
                    유코번 = 초중종번_조합번(초번, 중번, 종번)
                    자번 = 유코번 - 유코한글시작
                    자 = chr(유코번)

                    if 유코번 not in unicode_cmap.cmap:  # 없는 글자!
                        continue

                    img_hq = Image.new("RGB", (FONT_WIDTH * HQ_RATIO, FONT_HEIGHT * HQ_RATIO))
                    draw_hq = ImageDraw.Draw(img_hq)
                    draw_hq.text((0, 0), 자, font=font)
                    w, h = draw_hq.textsize(자, font=font)
                    assert w <= FONT_WIDTH * HQ_RATIO and h <= FONT_HEIGHT * HQ_RATIO

                    img = img_hq.resize((FONT_WIDTH, FONT_HEIGHT))
                    #
                    #
                    img_nd = np.sum(np.array(img), axis=2, dtype=np.bool)  # rgb to bw

                    if np.count_nonzero(img_nd) == 0:  # 없는 글자!
                        continue

                    if SHOW_FONT and 자번 in 체크번s:
                        print(font_fp, 자, w, h)
                        show_font(img_nd)

                    n_chrs += 1

                    한글_BMPs.append(img_nd[:, :])
                    한글_codes.append(자번)

                    #
                    for tfi in range(1, n_distorts):

                        tf_type = np.random.randint(3)

                        # 0. ratio
                        if tf_type == 0:
                            a = 0.75 + 0.5 * np.random.rand(2)
                            img_tf = Image.new("RGB", (FONT_WIDTH, FONT_HEIGHT), (0, 0, 0))
                            img_tf.paste(
                                img_hq.resize((int(FONT_WIDTH * a[0]), int(FONT_HEIGHT * a[1]))), (0, 0))
                        # 1. rotate
                        elif tf_type == 1:
                            a = 10 + np.random.rand() * 20
                            img_tf = img_hq.rotate(a)
                            img_tf = img_tf.resize((FONT_WIDTH, FONT_HEIGHT))
                        # 2. affine
                        else:
                            a = np.random.rand(4) * 0.3 - 0.15
                            img_tf = img_hq.transform(
                                (FONT_WIDTH * HQ_RATIO, FONT_HEIGHT * HQ_RATIO),
                                Image.AFFINE, (1.0+a[0], -a[1], 0, -a[2], 1+a[3], 0), Image.BILINEAR)
                            img_tf = img_tf.resize((FONT_WIDTH, FONT_HEIGHT))

                        img_tf_nd = np.sum(np.array(img_tf), axis=2, dtype=np.bool)

                        # white noise
                        N_WHITE_POINTS = 20
                        img_white = np.zeros((FONT_WIDTH, FONT_HEIGHT), dtype=np.bool)
                        white_pos = np.array(
                            np.random.random((N_WHITE_POINTS, 2)) * np.array([FONT_WIDTH, FONT_HEIGHT]),
                            dtype=np.int)
                        img_white[white_pos[:, 0], white_pos[:, 1]] = True
                        img_tf_nd += img_white

                        한글_BMPs.append(img_tf_nd[:, :])
                        한글_codes.append(자번)

                        if SHOW_FONT and 자번 in 체크번s:
                            print(font_fp, 자, w, h)
                            show_font(img_tf_nd)

        print("loaded {}/{} fonts * {} distortions".format(n_chrs, 한글총개수, max(1, n_distorts)))

    return 한글_BMPs, 한글_codes


#
def encode_onehot(vals, max_val):
    onehot = np.zeros((len(vals), max_val), dtype=np.bool)
    for i, val in enumerate(vals):
        onehot[i, val] = 1
    return onehot

#
train_inputs, train_codes = load_한글_BMPs(train_fonts, N_DISTORTIONS)
train_inputs = np.expand_dims(np.array(train_inputs, dtype=np.bool), 3)
train_codes_초, train_codes_중, train_codes_종 = 자번_분리번(np.array(train_codes))
train_labels_초 = encode_onehot(train_codes_초, 한글초성개수)
train_labels_중 = encode_onehot(train_codes_중, 한글중성개수)
train_labels_종 = encode_onehot(train_codes_종, 한글종성개수)

valid_inputs, valid_codes = load_한글_BMPs(valid_fonts, 0)
valid_inputs = np.expand_dims(np.array(valid_inputs, dtype=np.bool), 3)
valid_codes_초, valid_codes_중, valid_codes_종 = 자번_분리번(np.array(valid_codes))
valid_labels_초 = encode_onehot(valid_codes_초, 한글초성개수)
valid_labels_중 = encode_onehot(valid_codes_중, 한글중성개수)
valid_labels_종 = encode_onehot(valid_codes_종, 한글종성개수)


#
def fc_layer(prev_layer, n_nodes, activate=tf.nn.tanh, name=""):
    prev_n_nodes = prev_layer.get_shape().as_list()[-1]
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([prev_n_nodes, n_nodes], stddev=0.1, dtype=tf.float32),
                        name="weights")
        b = tf.Variable(tf.zeros([n_nodes], dtype=tf.float32), name="biases")

        z = tf.add(tf.matmul(prev_layer, w), b, name='zs')
        a = activate(z, name='as')

    return a

#
input_layer = tf.placeholder(tf.float32, [None, FONT_WIDTH, FONT_HEIGHT, 1], name='input')

#
phase_train = tf.placeholder(tf.bool, name='phase_train')


#
def conv_layer(prev_layer, kernel_size, n_filter, name):
    kernel_d = prev_layer.get_shape().as_list()[-1]

    with tf.name_scope(name):
        w = tf.Variable(
            tf.random_normal([kernel_size, kernel_size, kernel_d, n_filter],
                             stddev=0.1, dtype=tf.float32), name='weights')
        b = tf.Variable(
            tf.zeros([n_filter], dtype=tf.float32), name='biases')
        z = tf.add(
            tf.nn.conv2d(prev_layer, filter=w, strides=[1, 1, 1, 1], padding="SAME"),
            b, name='zs')
        a = tf.nn.elu(z, name='as')

    return a


#
def inception_layer(prev_layer, n_sub_reduced_depth, n_sub_filter, name):

    with tf.name_scope(name):

        sub_1x1_r = conv_layer(prev_layer, 1, n_sub_filter, name + "_sub_1x1_r")
        sub_3x3_r = conv_layer(prev_layer, 1, n_sub_reduced_depth, name + "_sub_3x3_r")
        sub_3x3_c = conv_layer(sub_3x3_r, 3, n_sub_filter, name + "_sub_3x3_c")
        sub_5x5_r = conv_layer(prev_layer, 1, n_sub_reduced_depth, name + "_sub_5x5_r")
        sub_5x5_c = conv_layer(sub_5x5_r, 5, n_sub_filter, name + "_sub_5x5_c")
        sub_max_m = max_pool_layer(prev_layer, 1, name + "_sub_max_m")
        sub_max_r = conv_layer(sub_max_m, 1, n_sub_filter, name + "_sub_max_r")

    return tf.concat(3, [sub_1x1_r, sub_3x3_c, sub_5x5_c, sub_max_r])


#
def conv_batch_normalize(layer, name):

    with tf.name_scope(name):
        n_filter = layer.get_shape().as_list()[-1]
        batch_mean, batch_var = tf.nn.moments(layer, [0, 1, 2])
        beta = tf.Variable(tf.zeros(n_filter, dtype=tf.float32))
        gamma = tf.Variable(tf.ones(n_filter, dtype=tf.float32))
        ema = tf.train.ExponentialMovingAverage(decay=BATCH_NORMALIZATION_DECAY)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        bn = tf.nn.batch_normalization(layer, mean, var, beta, gamma, 1e-4, name='bn')

    return bn


#
def max_pool_layer(prev_layer, size, name):
    with tf.name_scope(name):
        return tf.nn.max_pool(
            prev_layer, ksize=[1, size, size, 1], strides=[1, size, size, 1],
            padding="SAME", name='as')


#
def avg_pool_layer(prev_layer, stride, name):
    with tf.name_scope(name):
        return tf.nn.avg_pool(
            prev_layer, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],
            padding="SAME", name='as')


#
def flatten(prev_layer, name):
    with tf.name_scope(name):
        length = np.prod(prev_layer.get_shape().as_list()[1:])
        f = tf.reshape(prev_layer, [-1, length], name='flatten')

    return f

#
hidden_layer_0 = conv_layer(input_layer, 7, 64, name='hidden_0')
hidden_layer_0_bn = conv_batch_normalize(hidden_layer_0, name='hidden_0_bn')
hidden_layer_0_m = max_pool_layer(hidden_layer_0_bn, 2, name='hidden_0_m')
#
hidden_layer_1a = inception_layer(hidden_layer_0_m, 32, 32, name='hidden_1a')
hidden_layer_1b = inception_layer(hidden_layer_1a, 32, 32, name='hidden_1b')
hidden_layer_1c = inception_layer(hidden_layer_1b, 32, 32, name='hidden_1c')
hidden_layer_1_bn = conv_batch_normalize(hidden_layer_1c, name='hidden_1_bn')
# hidden_layer_1_sc = tf.concat(3, [hidden_layer_0_m, hidden_layer_1_bn])
hidden_layer_1_m = max_pool_layer(hidden_layer_1_bn, 2, name='hidden_1_m')
#
hidden_layer_2a = inception_layer(hidden_layer_1_m, 48, 48, name='hidden_2a')
hidden_layer_2b = inception_layer(hidden_layer_2a, 48, 48, name='hidden_2b')
hidden_layer_2c = inception_layer(hidden_layer_2b, 48, 48, name='hidden_2c')
hidden_layer_2_bn = conv_batch_normalize(hidden_layer_2c, name='hidden_2_bn')
# hidden_layer_2_sc = tf.concat(3, [hidden_layer_1_m, hidden_layer_2_bn])
hidden_layer_2_m = max_pool_layer(hidden_layer_2_bn, 2, name='hidden_2_m')
#
hidden_layer_3a = inception_layer(hidden_layer_2_m, 64, 64, name='hidden_3a')
hidden_layer_3b = inception_layer(hidden_layer_3a, 64, 64, name='hidden_3b')
hidden_layer_3c = inception_layer(hidden_layer_3b, 64, 64, name='hidden_3c')
hidden_layer_3_bn = conv_batch_normalize(hidden_layer_3c, name='hidden_3_bn')
# hidden_layer_3_sc = tf.concat(3, [hidden_layer_2_m, hidden_layer_3_bn])
hidden_layer_3_m = max_pool_layer(hidden_layer_3_bn, 2, name='hidden_3_m')
#
hidden_layer_4a = inception_layer(hidden_layer_3_m, 96, 96, name='hidden_4a')
hidden_layer_4b = inception_layer(hidden_layer_4a, 96, 96, name='hidden_4b')
hidden_layer_4c = inception_layer(hidden_layer_4b, 96, 96, name='hidden_4c')
hidden_layer_4_bn = conv_batch_normalize(hidden_layer_4c, name='hidden_4_bn')
# hidden_layer_4_sc = tf.concat(3, [hidden_layer_3_m, hidden_layer_4_bn])
hidden_layer_4_a = avg_pool_layer(hidden_layer_4_bn, 1, name='hidden_4_a')
#
hidden_layer_5 = flatten(hidden_layer_4_a, name='hidden_5')
hidden_layer_6 = fc_layer(hidden_layer_5, 512, name='hidden_6')
hidden_layer_last = fc_layer(hidden_layer_6, 512, name='hidden_last')

output_layer_초 = fc_layer(hidden_layer_last, 한글초성개수, activate=tf.nn.elu, name='output_i')
output_layer_중 = fc_layer(hidden_layer_last, 한글중성개수, activate=tf.nn.elu, name='output_m')
output_layer_종 = fc_layer(hidden_layer_last, 한글종성개수, activate=tf.nn.elu, name='output_f')

output_layer = tf.cast(tf.argmax(output_layer_초, 1) * 한글중성개수 * 한글종성개수 +
                       tf.argmax(output_layer_중, 1) * 한글종성개수 +
                       tf.argmax(output_layer_종, 1), tf.int32, name='output_layer')
#
output_label_초 = tf.placeholder(tf.float32, [None, 한글초성개수], name='output_i')
output_label_중 = tf.placeholder(tf.float32, [None, 한글중성개수], name='output_m')
output_label_종 = tf.placeholder(tf.float32, [None, 한글종성개수], name='output_m')
#
output_code = tf.placeholder(tf.int32, [None], name='output_code')
#
loss_초 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(output_layer_초, output_label_초, name='loss_i'))
loss_중 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(output_layer_중, output_label_중, name='loss_m'))
loss_종 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(output_layer_종, output_label_종, name='loss_f'))
loss_총 = loss_초 + loss_중 + loss_종
#
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_총)
#
correct_prediction = tf.equal(output_code, output_layer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
saver = tf.train.Saver()
#
sess = tf.Session()

#
ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
if LOAD_MODEL and ckpt and ckpt.model_checkpoint_path:
    new_saver = tf.train.import_meta_graph(MODEL_DIR + '/' + 'model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR + '/'))
    print("Loaded model")
else:
    init = tf.global_variables_initializer()
    sess.run(init, feed_dict={phase_train: True})


def feeds(set_str, n_batch=1, batch=0):

    if set_str == "train":
        inputs = train_inputs
        labels_초 = train_labels_초
        labels_중 = train_labels_중
        labels_종 = train_labels_종
        codes = train_codes
    else:
        assert set_str == 'valid'
        inputs = valid_inputs
        labels_초 = valid_labels_초
        labels_중 = valid_labels_중
        labels_종 = valid_labels_종
        codes = valid_codes

    batch_start = int(len(inputs) / n_batch) * batch
    next_batch_start = int(len(inputs) / n_batch) * (batch + 1)
    if ONLY_ONE_EXAMPLE_PER_MINI_BATCH:
        batch_ixs = slice(batch_start, batch_start + 1, 1)
    else:
        batch_ixs = slice(batch_start, next_batch_start, 1)

    return {
        phase_train: True if set_str == "train" else False,
        input_layer: inputs[batch_ixs],
        output_label_초: labels_초[batch_ixs],
        output_label_중: labels_중[batch_ixs],
        output_label_종: labels_종[batch_ixs],
        output_code: codes[batch_ixs],
    }


def float_formatter(x): return "{:9.5f}".format(x)
np.set_printoptions(formatter={'float_kind': float_formatter})

print("start training...")
for ep in range(TOTAL_EPOCHS):

    my_train_n_batch = int(math.ceil(len(train_inputs) / MINI_BATCH_SIZE))
    my_valid_n_batch = int(math.ceil(len(valid_inputs) / MINI_BATCH_SIZE))

    result = np.zeros((my_train_n_batch, 6))
    valid_acc = np.zeros(my_valid_n_batch)
    for my_batch in range(my_train_n_batch):
        result[my_batch, :] = sess.run([accuracy, loss_총, loss_초, loss_중, loss_종, optimizer],
                                       feed_dict=feeds("train", my_train_n_batch, my_batch))
        if my_batch < my_valid_n_batch and ep % SHOW_STATUS_PER_EPOCH == 0:
            valid_acc[my_batch], = sess.run(
                [accuracy], feed_dict=feeds("valid", my_valid_n_batch, my_batch))
    if ep % SHOW_STATUS_PER_EPOCH == 0:
        print("end of epoch:{:>6}, train_acc:{:7.3f}%, valid_acc:{:7.3f}%, loss:{}".format(
            ep, result[:, 0].mean(axis=0) * 100,
            valid_acc.mean() * 100, np.array(result[:, 1:-1].mean(axis=0))))
    if ep > 0 and ep % SAVE_MODEL_PER_EPOCHS == 0:
        saver.save(sess, MODEL_DIR + "/" + 'model')
        print("Saved model")