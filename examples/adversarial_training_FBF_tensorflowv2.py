"""
This is an example of how to use ART for adversarial training of a model with Fast is better than free protocol
"""

from art.data_generators import TensorFlowV2DataGenerator
from art.defences.trainer import AdversarialTrainerFBFTensorflowv2
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_cifar10

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D

"""
For this example we choose the PreActResNet model as used in the paper (https://openreview.net/forum?id=BJx040EFvH)
The code for the model architecture has been adopted from
https://github.com/anonymous-sushi-armadillo/fast_is_better_than_free_CIFAR10/blob/master/preact_resnet.py
"""


class TensorFlowModel(Model):
    """
    Standard TensorFlow model for unit testing.
    """

    def __init__(self):
        super(TensorFlowModel, self).__init__()
        self.conv1 = Conv2D(filters=4, kernel_size=5, activation='relu')
        self.conv2 = Conv2D(filters=10, kernel_size=5, activation='relu')
        self.maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format=None)
        self.flatten = Flatten()
        self.dense1 = Dense(100, activation='relu')
        self.logits = Dense(10, activation='linear')

    def call(self, x):
        """
        Call function to evaluate the model.

        :param x: Input to the model
        :return: Prediction of the model
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.logits(x)
        return x


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients = [(tf.clip_by_global_norm(grad, 0.5))
                 for grad in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# Step 1: Load the CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# prepare the tensors for preprocessing
cifar_mu = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2471, 0.2435, 0.2616)

# Step 2: create the PyTorch model

model = TensorFlowModel()
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
