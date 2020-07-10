"""
This is an example of how to use ART for adversarial training of a model with Fast is better than free protocol
"""

import numpy as np

from art.data_generators import TensorFlowV2DataGenerator
from art.defences.trainer import AdversarialTrainerFBFTensorflowv2
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_cifar10
from art.attacks.evasion import ProjectedGradientDescent

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

"""
For this example we choose the PreActResNet model as used in the paper (https://openreview.net/forum?id=BJx040EFvH)
The code for the model architecture has been adopted from
https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/preact_resnet.py
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model


class BasicBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BasicBlock, self).__init__()

        self.features = Sequential([
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(kernels, (3, 3), strides=1, padding='same'),
            layers.BatchNormalization()
        ])

        if stride != 1:
            shortcut = [
                layers.Conv2D(kernels, (1, 1), strides=stride),
                layers.BatchNormalization()
            ]
        else:
            shortcut = []
        self.shorcut = Sequential(shortcut)

    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)
        x = tf.nn.relu(layers.add([residual, x]))
        return x


class BottleNeckBlock(layers.Layer):
    def __init__(self, kernels, stride=1):
        super(BottleNeckBlock, self).__init__()

        self.features = Sequential([
            layers.Conv2D(kernels, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels, (3, 3), strides=stride, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(kernels * 4, (1, 1), strides=1, padding='same'),
            layers.BatchNormalization(),
        ])

        self.shorcut = Sequential([
            layers.Conv2D(kernels * 4, (1, 1), strides=stride),
            layers.BatchNormalization()
        ])

    def call(self, inputs, training=False):
        residual = self.shorcut(inputs, training=training)
        x = self.features(inputs, training=training)
        x = tf.nn.relu(x + residual)
        return x


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes, input_shape=(32, 32, 3)):
        super(ResNet, self).__init__()
        self.conv1 = Sequential([
            layers.Input(input_shape),
            layers.Conv2D(64, (3, 3), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.conv2_x = self._make_layer(block, 64, num_blocks[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_blocks[3], 2)
        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation='softmax')

    def _make_layer(self, block, kernels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        nets = []
        for stride in strides:
            nets.append(block(kernels, stride))
        return Sequential(nets)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.gap(x)
        x = self.fc(x)
        return x


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes):
    return ResNet(BottleNeckBlock, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes):
    return ResNet(BottleNeckBlock, [3, 8, 36, 3], num_classes)

# class TensorFlowModel(Model):
#
#     def __init__(self):
#         super(TensorFlowModel, self).__init__()
#         self.conv1 = Conv2D(filters=4, kernel_size=5, activation='relu')
#         self.conv2 = Conv2D(filters=10, kernel_size=5, activation='relu')
#         self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)
#         self.flatten = Flatten()
#         self.dense1 = Dense(100, activation='relu')
#         self.logits = Dense(10, activation='linear')
#
#     def call(self, x):
#         """
#         Call function to evaluate the model.
#
#         :param x: Input to the model
#         :return: Prediction of the model
#         """
#         x = self.conv1(x)
#         x = self.maxpool(x)
#         x = self.conv2(x)
#         x = self.maxpool(x)
#         x = self.flatten(x)
#         x = self.dense1(x)
#         x = self.logits(x)
#         return x
#

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    # gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Step 1: Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# prepare the tensors for preprocessing
cifar_mu = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

# Step 2: create the PyTorch model

model = ResNet18(10)
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(lambda x, y:
                                  (tf.image.random_flip_left_right(x), y)
                                  ).shuffle(10000).batch(128).repeat()

# Step 3: Create the ART classifier
classifier = TensorFlowV2Classifier(model=model, loss_object=loss_object,
                                    preprocessing=(cifar_mu, cifar_std),
                                    train_step=train_step, nb_classes=10,
                                    input_shape=(32, 32, 3), clip_values=(0, 1))

# Step 4: Create the trainer object - AdversarialTrainerFBFPyTorch
# if you have apex installed, change use_amp to True
epsilon = (8.0 / 255.)
trainer = AdversarialTrainerFBFTensorflowv2(classifier, eps=epsilon)

art_datagen = TensorFlowV2DataGenerator(iterator=train_dataset,
                                        size=x_train.shape[0],
                                        batch_size=128
                                        )
# Step 5: fit the trainer
trainer.fit_generator(art_datagen, nb_epochs=1)

x_test_pred = np.argmax(trainer.predict(x_test), axis=1)
print(
    "Accuracy on original PGD adversarial samples: %.2f%%"
    % (np.sum(x_test_pred == np.argmax(y_test, axis=1))
       / x_test.shape[0] * 100)
)

attack = ProjectedGradientDescent(
    classifier,
    norm=np.inf,
    eps=8.0 / 255.0,
    eps_step=2.0 / 255.0,
    max_iter=40,
    targeted=False,
    num_random_init=5,
    batch_size=32,
)
x_test_attack = attack.generate(x_test)
x_test_attack_pred = np.argmax(trainer.predict(x_test_attack), axis=1)
print(
    "Accuracy on original PGD adversarial samples: %.2f%%"
    % (np.sum(x_test_attack_pred == np.argmax(y_test, axis=1))
       / x_test.shape[0] * 100)
)
