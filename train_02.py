import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers

from sklearn.model_selection import KFold

import numpy as np
import os
import random, re, math
import matplotlib.pyplot as plt

import efficientnet.tfkeras as efn
import keras

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print("Tensorflow version " + tf.__version__)

IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU
EPOCHS = 15
BATCH_SIZE = 4

NUM_TRAINING_IMAGES = 21397
NUM_TEST_IMAGES = 1
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

FILENAMES = tf.io.gfile.glob('/home/hong/dl_data/cassava_leaf_disease_classification/train_resize_tfrecords/ld_train*.tfrec')

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])  # explicit size needed for TPU

    return image


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    label = tf.cast(example['target'], tf.int32)
    image = decode_image(example['image'])

    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum  # returns a dataset of image(s)


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                              [3, 3])

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))


def transform(image, label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SIZE[0]
    XDIM = DIM % 2  # fix for size 331

    image = tf.image.random_crop(image, [480, 480, 3])
    image = tf.image.resize(image, [DIM, DIM])

    rot = 15. * tf.random.normal([1], dtype='float32')
    shr = 5. * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    h_shift = 16. * tf.random.normal([1], dtype='float32')
    w_shift = 16. * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3]), label


def load_dataset(filenames, labeled=True, ordered=False, is_train=True):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order)  # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset


def get_training_dataset(filenames):
    dataset = load_dataset(filenames, labeled=True, is_train=True)

    # train data augmentation
    dataset = dataset.map(transform)

    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset


def get_validation_dataset(filenames):
    dataset = load_dataset(filenames, labeled=True, ordered=False, is_train=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    return dataset


# Using an LR ramp up because fine-tuning a pre-trained model.
# Starting with a high LR would break the pre-trained weights.

LR_START = 0.00001
LR_MAX = 0.00005
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 5
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr


learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(25 if EPOCHS < 25 else EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


skf = KFold(n_splits=5, shuffle=True, random_state=2020)

results_list = []

for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(FILENAMES)))):
    print(f'Fold ::::: {fold}')

    filenames_tr = [FILENAMES[i] for i in train_idx]
    filenames_val = [FILENAMES[i] for i in val_idx]

    data_tr = get_training_dataset(filenames_tr)
    data_val = get_validation_dataset(filenames_val)

    # include_top=False : without last fcn
    base_model = efn.EfficientNetB4(weights='noisy-student', include_top=False, input_shape=[*IMAGE_SIZE, 3])

    # Unfreeze the base model
    base_model.trainable = True

    model1 = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    # default lr : 0.01
    opt = optimizers.Adam(lr=0.01)

    model1.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        # loss = [focal_loss],
        metrics=['sparse_categorical_accuracy']
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint('ckpt/effB4-crop_aug-fold-%i.h5' % fold,
                                                    monitor='val_sparse_categorical_accuracy', verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True, save_freq='epoch')

    historical1 = model1.fit(data_tr,
                             steps_per_epoch=STEPS_PER_EPOCH,
                             epochs=EPOCHS,
                             callbacks=[learning_rate_callback, checkpoint],
                             validation_data=data_val)

    model1.load_weights('ckpt/effB3-fold-%i.h5' % fold)

    results = model1.evaluate(data_val)
    results_list.append(results[1])


print(f'Result ::: {results_list}')
print(f'Result ::: {np.mean(results_list)}')