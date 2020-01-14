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
#
# %%
from tensorflow import keras
import numpy as np

imdb = keras.datasets.imdb
num_words = 99999

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=num_words)
print(train_data[0])

# %%
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNKNOWN>"] = 2
word_index["<UNUSED>"] = 3
print(word_index)

# %%
reverse_word_index = zip(word_index.values(), word_index.keys())
reverse_word_index = dict(reverse_word_index)
print(type(reverse_word_index))

# %%
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding='post',
    maxlen=256
)
print('ok'.center(90, '|'))


# %%
def decode_review(text):
    return ' '.join([reverse_word_index.get(_, '?') for _ in text])


print(decode_review(test_data[0]))

x_val = train_data[:num_words // 10]
x_train = train_data[num_words // 10 + 1:]

y_val = train_labels[:num_words // 10]
y_train = train_labels[num_words // 10 + 1:]
print('ok'.center(90, '|'))

# %%

model = keras.Sequential((
    keras.layers.Embedding(num_words, 16),
    keras.layers.GlobalAvgPool1D(),
    keras.layers.Dense(16, activation=keras.activations.relu),
    keras.layers.Dense(1, activation=keras.activations.sigmoid),
))

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam.__name__,
    loss=keras.losses.binary_crossentropy.__name__,
    metrics=['acc']
)
print('ok'.center(90, '|'))

# %%
fit_model = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=512,
    validation_data=(x_val, y_val)
)
# Saving models
model.save('imdb_model')
print('ok'.center(90, '|'))

# %%
result = model.evaluate(test_data, test_labels)
print(result)
print('ok'.center(90, '|'))

# %%
# Loading models
model = keras.models.load_model('imdb_model')


# %%
def review_encode(text):
    _encoded = [1]
    for word in text:
        word = word.strip().lower()
        if not word:
            continue
        if word in word_index:
            _encoded.append(word_index[word])
        else:
            _encoded.append(2)
    return _encoded


with open('test.txt') as f:
    while ...:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        print(line)
        nline = line.replace(',', '') \
            .replace('.', '') \
            .replace('(', '') \
            .replace(')', '') \
            .replace(';', '') \
            .replace(':', '') \
            .replace(':', '') \
            .strip() \
            .split(' ')
        if not line:
            continue
        encoded = review_encode(nline)
        encoded = keras.preprocessing.sequence.pad_sequences(
            [encoded],
            value=word_index["<PAD>"],
            padding='post',
            maxlen=256
        )
        predict = model.predict(encoded)
        # print(encoded)
        print(predict[0])
        print('ok'.center(90, '`'))