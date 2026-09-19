import tensorflow as tf
from tensorflow.keras import layers, Model


def convolution_block(inputs, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    return x


def build_unet(input_shape=(128, 128, 3)):

    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = convolution_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = convolution_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = convolution_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    c4 = convolution_block(p3, 256)

    # Decoder
    u1 = layers.UpSampling2D()(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = convolution_block(u1, 128)

    u2 = layers.UpSampling2D()(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = convolution_block(u2, 64)

    u3 = layers.UpSampling2D()(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = convolution_block(u3, 32)

    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid"
    )(c7)

    model = Model(
        inputs,
        outputs
    )

    return model