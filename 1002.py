# -*- coding: utf-8 -*-

#
import numpy as np
import matplotlib.pyplot as plt

#
from PIL import Image, ImageFont, ImageDraw
from fontTools.ttLib import TTFont

#
from keras.models import Sequential
from keras.layers.core import Reshape
from keras.layers import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras import optimizers

FONT_DIR = "fonts"

SOURCE_FONT_FN = "gulim.ttc"
TARGET_FONT_FN = "NanumMyeongjo.ttf"

FONT_SIZE = 32
FONT_WIDTH = 40
FONT_HEIGHT = 40
FONT_LENGTH = FONT_WIDTH * FONT_HEIGHT

NUM_HANGULS = 19*21*28
HG_UC_START = 0xac00
HG_UC_END = HG_UC_START + NUM_HANGULS


def show_font(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none')
    plt.draw()
    plt.show(block=False)
    plt.waitforbuttonpress()


def load_hangul_imgs(font_fn):

    # CHECK_CNS = [0, 100, 1000, 7000, 9000]
    CHECK_CNS = []

    hangul_imgs = np.zeros((NUM_HANGULS, FONT_WIDTH, FONT_HEIGHT, 1), dtype=np.float32)
    hangul_loaded = np.zeros(NUM_HANGULS, dtype=np.bool)

    font_fp = FONT_DIR + "/" + font_fn

    font = ImageFont.truetype(font_fp, FONT_SIZE)
    ft = TTFont(font_fp, fontNumber=0)

    # cmap: https://www.microsoft.com/typography/otspec/cmap.htm
    unicode_cmap = ft['cmap'].getcmap(3, 1)

    for uc in range(HG_UC_START, HG_UC_END+1):

        c = chr(uc)
        cn = uc - HG_UC_START

        if uc not in unicode_cmap.cmap:  # 없는 글자!
            continue

        img = Image.new("RGB", (FONT_WIDTH, FONT_HEIGHT))
        drawer = ImageDraw.Draw(img)
        drawer.text((0, 0), c, font=font)
        w, h = drawer.textsize(c, font=font)
        assert w <= FONT_WIDTH and h <= FONT_HEIGHT

        #
        img_nd = np.mean(np.array(img), axis=2, dtype=np.float32)/255.0  # rgb to gray
        img_nd = np.expand_dims(img_nd, 2)

        if np.count_nonzero(img_nd) == 0:  # 없는 글자!
            continue

        if cn in CHECK_CNS:
            print("showing...", font_fp, c, w, h)
            show_font(img_nd)

        hangul_imgs[cn, :] = img_nd
        hangul_loaded[cn] = True

    return hangul_imgs, hangul_loaded


s_imgs, s_loaded = load_hangul_imgs(SOURCE_FONT_FN)
t_imgs, t_loaded = load_hangul_imgs(TARGET_FONT_FN)
both_loaded = s_loaded & t_loaded

#
model = Sequential()
model.add(Conv2D(128, kernel_size=(5, 5), padding='same',
                 input_shape=[FONT_WIDTH, FONT_HEIGHT, 1]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2DTranspose(256, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2DTranspose(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
model.add(Activation('relu'))
model.add(Conv2DTranspose(1, kernel_size=(5, 5), padding='same'))
model.add(Activation('relu'))

adam = optimizers.adam(lr=1e-4)
model.compile(optimizer=adam, loss='mse')

#
x_train = s_imgs[both_loaded]
y_train = t_imgs[both_loaded]
x_test = s_imgs[both_loaded]

#
model.fit(x_train, y_train, epochs=1000, batch_size=128)
y_test = model.predict(x_test)

#
for i in range(len(y_test)):
    show_font(y_test[i].squeeze())

