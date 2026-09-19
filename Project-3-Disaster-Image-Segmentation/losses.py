import tensorflow as tf


def focal_loss(
    y_true,
    y_pred,
    alpha=0.25,
    gamma=2.0
):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.clip_by_value(
        y_pred,
        1e-7,
        1.0 - 1e-7
    )

    loss = (
        -alpha
        * y_true
        * tf.pow(
            1 - y_pred,
            gamma
        )
        * tf.math.log(y_pred)
    )

    return tf.reduce_mean(loss)