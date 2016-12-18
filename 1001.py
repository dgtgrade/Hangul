import sys
import random
import math
# import time
import numpy as np
import tensorflow as tf

#
from PIL import Image, ImageFont, ImageDraw

# import matplotlib.pyplot as plt
#
ONLY_ONE_EXAMPLE_PER_MINI_BATCH = True
#
VALID_RATIO = 0.05
MINI_BATCH_SIZE = 1000
# MODEL = "ALL_FC"
MODEL = "CONV"
SHOW_STATUS_PER_EPOCH = 1
LEARNING_RATE = 1 * 1e-4
N_FONTS = 1  # None for All
BATCH_NORMALIZATION_DECAY = 0.95
TOTAL_EPOCHS = 10000

#
# 글꼴들
FONT_DIR = "fonts"
fonts = [
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
    "HGSS.ttf"
    # http://www.hoondesign.co.kr/sub/hoonfont.php
    "HoonWhitecatR.ttf"
    # http://drstyle.blog.me/30158806258
    "drfont_daraehand_Basic.ttf"
]
#
if N_FONTS is None:
    N_FONTS = len(fonts)

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


def load_한글_BMPs(n_fonts=N_FONTS):
    한글_BMPs = np.zeros((n_fonts, 한글총개수, FONT_WIDTH, FONT_HEIGHT),
                       dtype=np.float16)
    #
    for font_i, font_fps in enumerate(fonts[0:n_fonts]):

        font = ImageFont.truetype(FONT_DIR + "/" + font_fps, FONT_SIZE)
        n_na = 0

        #
        # 72*72*10000*4B <= 300MB
        #
        print("loading... {}".format(font_fps))
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
                    imgnd = np.mean(np.array(img), axis=2, dtype=np.bool)  # rgb to gray
                    if np.count_nonzero(imgnd) == 0:
                        n_na += 1
                    한글_BMPs[font_i, 자번, :, :] = imgnd[:, :]

                    # if 자번 in [0, 7100, 9000]:
                    #     print(font_fps, 자, w, h)
                    #     plt.imshow(한글BMPs[font_i, 자번, :, :], cmap='gray', vmin=0, vmax=1, interpolation='none')
                    #     plt.draw()
                    #     plt.show(block=False)
                    #     plt.waitforbuttonpress()
        print("loaded {}/{} fonts".format(한글총개수-n_na, 한글총개수))
    return 한글_BMPs

#
codes = np.tile(np.arange(한글총개수), [N_FONTS])
codes_초중종 = np.tile(np.array([자번_분리번(자번) for 자번 in range(한글총개수)]), [N_FONTS, 1])
labels_초 = np.tile(np.zeros((한글총개수, 한글초성개수), dtype=np.bool), [N_FONTS, 1])
labels_중 = np.tile(np.zeros((한글총개수, 한글중성개수), dtype=np.bool), [N_FONTS, 1])
labels_종 = np.tile(np.zeros((한글총개수, 한글종성개수), dtype=np.bool), [N_FONTS, 1])
labels_초[np.arange(N_FONTS * 한글총개수), codes_초중종[:, 0]] = True
labels_중[np.arange(N_FONTS * 한글총개수), codes_초중종[:, 1]] = True
labels_종[np.arange(N_FONTS * 한글총개수), codes_초중종[:, 2]] = True
# print(labels_초)
# print(labels_중)
# print(labels_종)

my_한글_BMPs = load_한글_BMPs(n_fonts=N_FONTS)
my_한글_NAs = -(my_한글_BMPs.sum(-1).sum(-1).astype(np.bool))


#
def fc_layer(prev_layer, n_nodes, activate=tf.nn.tanh, name=""):
    prev_n_nodes = prev_layer.get_shape().as_list()[-1]
    with tf.name_scope(name):
        w = tf.Variable(tf.random_normal([prev_n_nodes, n_nodes], stddev=0.1, dtype=tf.float32), name="weights")
        b = tf.Variable(tf.zeros([n_nodes], dtype=tf.float32), name="biases")

        z = tf.add(tf.matmul(prev_layer, w), b, name='zs')
        a = activate(z, name='as')

    return a


#
if MODEL == "ALL_FC":
    #
    inputs = my_한글_BMPs.reshape(N_FONTS * 한글총개수, FONT_LENGTH).astype(np.float32)
    #
    input_layer = tf.placeholder(tf.float32, [None, FONT_LENGTH], name='input')
    #
    hidden_layer_0 = fc_layer(input_layer, 512, name='hidden_0')
    hidden_layer_1 = fc_layer(hidden_layer_0, 512, name='hidden_1')
    hidden_layer_2 = fc_layer(hidden_layer_1, 512, name='hidden_2')
    hidden_layer_3 = fc_layer(hidden_layer_2, 512, name='hidden_3')
    hidden_layer_last = fc_layer(hidden_layer_3, 512, name='hidden_last')
    #
    init_feeds = {}
    #
elif MODEL == "CONV":
    inputs = my_한글_BMPs.reshape(N_FONTS * 한글총개수, FONT_WIDTH, FONT_HEIGHT, 1).astype(np.float32)
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
    
            sub_1x1_r = conv_layer(prev_layer, 1, n_sub_reduced_depth, name + "_sub_1x1_r")
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
    def avg_pool_layer(prev_layer, name):
        with tf.name_scope(name):
            return tf.nn.avg_pool(
                prev_layer, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding="SAME", name='as')

    #
    def flatten(prev_layer, name):
        with tf.name_scope(name):
            length = np.prod(prev_layer.get_shape().as_list()[1:])
            f = tf.reshape(prev_layer, [-1, length], name='flatten')

        return f

    #
    hidden_layer_0 = conv_layer(input_layer, 3, 64, name='hidden_0')
    hidden_layer_0_bn = conv_batch_normalize(hidden_layer_0, name='hidden_0_bn')
    #
    hidden_layer_1 = inception_layer(input_layer, 32, 32, name='hidden_1')
    hidden_layer_1_bn = conv_batch_normalize(hidden_layer_1, name='hidden_1_bn')
    #
    hidden_layer_2 = inception_layer(input_layer, 32, 64, name='hidden_2')
    hidden_layer_2_bn = conv_batch_normalize(hidden_layer_1, name='hidden_2_bn')
    #
    hidden_layer_3 = inception_layer(input_layer, 32, 128, name='hidden_3')
    hidden_layer_3_bn = conv_batch_normalize(hidden_layer_1, name='hidden_3_bn')
    #
    hidden_layer_4_0 = avg_pool_layer(hidden_layer_3_bn, name='hidden_4')
    #
    hidden_layer_5 = flatten(hidden_layer_4_0, name='hidden_5')
    hidden_layer_6 = fc_layer(hidden_layer_5, 512, name='hidden_6')
    hidden_layer_last = fc_layer(hidden_layer_6, 512, name='hidden_last')

    init_feeds = {phase_train: True}

else:
    sys.exit(1)

#

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
loss_초 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_초, output_label_초, name='loss_i'))
loss_중 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_중, output_label_중, name='loss_m'))
loss_종 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output_layer_종, output_label_종, name='loss_f'))
loss_총 = loss_초 + loss_중 + loss_종
#
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss_총)
#
correct_prediction = tf.equal(output_code, output_layer)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

#
init = tf.global_variables_initializer()
sess.run(init, feed_dict=init_feeds)

#
valid_ixs = np.tile(np.random.random(한글총개수) < VALID_RATIO, [N_FONTS])
train_ixs = -valid_ixs
na_ixs = my_한글_NAs.reshape(N_FONTS*한글총개수)

train_ixs = np.array((train_ixs & (-na_ixs)).nonzero()[0])
valid_ixs = np.array((valid_ixs & (-na_ixs)).nonzero()[0])


def feeds(set_str, n_batch=1, batch=0):
    ixs = {
        "train": train_ixs,
        "valid": valid_ixs
    }[set_str]

    batch_start = int(len(ixs) / n_batch) * batch
    next_batch_start = int(len(ixs) / n_batch) * (batch + 1)
    if ONLY_ONE_EXAMPLE_PER_MINI_BATCH:
        batch_ixs = ixs[batch_start:batch_start + 1]
    else:
        batch_ixs = ixs[batch_start:next_batch_start]

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

    my_train_n_batch = int(math.ceil(len(train_ixs) / MINI_BATCH_SIZE))
    my_valid_n_batch = int(math.ceil(len(valid_ixs) / MINI_BATCH_SIZE))

    result = np.zeros((my_train_n_batch, 6))
    valid_acc = np.zeros(my_valid_n_batch)
    for my_batch in range(my_train_n_batch):
        result[my_batch, :] = \
            sess.run([accuracy, loss_총, loss_초, loss_중, loss_종, optimizer],
                     feed_dict=feeds("train", my_train_n_batch, my_batch))
        if my_batch < my_valid_n_batch and ep % SHOW_STATUS_PER_EPOCH == 0:
            valid_acc[my_batch], = sess.run(
                [accuracy], feed_dict=feeds("valid", my_valid_n_batch, my_batch))
    # noinspection PyUnboundLocalVariable
    if ep % SHOW_STATUS_PER_EPOCH == 0:
        print("end of epoch:{:>6}, train_acc:{:7.3f}%, valid_acc:{:7.3f}%, loss:{}".format(
            ep, result[:, 0].mean(axis=0) * 100, valid_acc.mean() * 100, np.array(result[:, 1:-1].mean(axis=0))))
