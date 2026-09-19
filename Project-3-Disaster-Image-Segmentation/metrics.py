import tensorflow as tf


def dice_coefficient(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    return (
        (2.0 * intersection + 1.0)
        /
        (
            tf.reduce_sum(y_true)
            +
            tf.reduce_sum(y_pred)
            +
            1.0
        )
    )


def iou_score(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(
        y_true * y_pred
    )

    union = (
        tf.reduce_sum(y_true)
        +
        tf.reduce_sum(y_pred)
        -
        intersection
    )

    return (
        (intersection + 1.0)
        /
        (union + 1.0)
    )