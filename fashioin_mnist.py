# %%
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Warning and Error only

import tensorflow as tf
from tensorflow.python.client import device_lib
import re

print(f'Tensorflow v{tf.__version__}'.center(90, '·'))

tf_exp_config = tf.config.experimental

print('Local Devices:')
gpu_name_re = re.compile('name: ?([A-Za-z0-9 ]+)(?=(?:,|$))')
cp_cap_re = re.compile('compute capability: ?([0-9.]+)(?=(?:,|$))')
unit = ('M', 'G', 'T', 'P''E',)
for i, x in enumerate(device_lib.list_local_devices()):
    print(f'\t{i}: name \t= [{x.name}]')
    try:
        _desc = x.physical_device_desc
        print(f'\t{i}: model\t= [{gpu_name_re.search(_desc).group(1)}]')
        print(f'\t{i}: capablt.\t= [{cp_cap_re.search(_desc).group(1)}]')
        del _desc
    except AttributeError:
        ...
    print(f'\t{i}: type \t= [{x.device_type}]')
    _mem = x.memory_limit / 1024 / 1024
    unit_i = 0
    while _mem > 1024 and unit_i < len(unit):
        _mem /= 1024
        unit_i += 1
    print(f'\t{i}: memory\t= [{_mem:.1f} {unit[unit_i]}iB]')
    print()
    del _mem, unit_i
del gpu_name_re

print('Visible Devices:')
for i, x in enumerate(tf_exp_config.get_visible_devices()):
    print(f'\t{i}: {x}')
print()

print(f'Device Policy: {tf_exp_config.get_device_policy()}')
print()

print(f'Visible GPU Configs:')
gpus = tf_exp_config.get_visible_devices('GPU')
for i, x in enumerate(gpus):
    print(f'\t{i}: {x}')
    tf_exp_config.set_memory_growth(x, True)
    print(f'\t{i}: Memory Growth: {tf_exp_config.get_memory_growth(x)}')
print()

del tf_exp_config, unit

# %%
from tensorflow.keras.mixed_precision import experimental as mixed_precision

for i, x in enumerate(device_lib.list_local_devices()):
    if x.device_type != 'GPU':
        continue
    _desc = x.physical_device_desc
    _capability = float(cp_cap_re.search(_desc).group(1))
    if _capability > 7:
        print('Enabling Mixed Precision'.center(90, '·'))
        float16_policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(float16_policy)
        print(f'\tCompute  dtype: {float16_policy.compute_dtype}')
        print(f'\tVariable dtype: {float16_policy.variable_dtype}')
        del float16_policy
    else:
        print(f'Skipping Mixed Precision due to compute capability [{_capability}] < 7')
    del cp_cap_re, _desc, _capability
    print()
    break
del i, x

# %%
tf.config.optimizer.set_jit(True)
print('Enabled XLA for TensorFlow models'.center(90, '·'))

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# %%
# Download Dataset
import numpy as np

keras = tf.keras

fashion_mnist = keras.datasets.fashion_mnist
(train_imgs, train_lbls), (test_imgs, test_lbls) = fashion_mnist.load_data()
train_imgs.shape = (train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2], 1)
test_imgs.shape = (test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2], 1)
labels = [
    '0 T-shirt/top',
    '1 Trouser',
    '2 Pullover',
    '3 Dress',
    '4 Coat',
    '5 Sandal',
    '6 Shirt',
    '7 Sneaker',
    '8 Bag',
    '9 Ankle boot'
]
print(labels)

# %%
# Normalize
train_imgs = train_imgs / 255
test_imgs = test_imgs / 255
print(train_imgs[0])
print(type(train_imgs[0]))
print('Data Normalization'.center(90, '·'))

# %%
# Build Models
import os

if os.path.isdir('mnist_model'):
    model = keras.models.load_model('mnist_model')
    print('Model Loaded'.center(90, '·'))

else:
    model = keras.Sequential((
        keras.layers.Conv2D(2, kernel_size=3, input_shape=(28, 28, 1),
                            activation=keras.activations.relu.__name__),

        # keras.layers.Dense(128, activation=keras.activations.relu.__name__),
        # keras.layers.Dense(64, activation=keras.activations.relu.__name__),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
        keras.layers.Activation(keras.activations.softmax.__name__, dtype=tf.dtypes.float32)
    ))

    # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
    # losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name="sigmoid_losses")
    # l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
    #                      name="l2_losses") * l2_reg_lambda
    # self.loss = tf.add(losses, l2_losses, name="loss")

    model.compile(
        optimizer=keras.optimizers.RMSprop.__name__,
        loss=keras.losses.sparse_categorical_crossentropy.__name__,
        metrics=['acc']
    )
    model.summary()
    print('Model Compiled'.center(90, '·'))

# %%
# Fit Model
model.fit(train_imgs, train_lbls, batch_size=512, epochs=10)

model.save('mnist_model')
print('Model Fit'.center(90, '·'))

# %%
# Eval Test Result
test_loss, test_acc = model.evaluate(test_imgs, test_lbls)
print(test_loss, test_acc)

# %%
# Prediction
import numpy as np

predict = model.predict(test_imgs)
np.argmax(predict[0])
# print(tf.argmax(predict[0]).numpy())
# print(test_lbls[0])

# %%