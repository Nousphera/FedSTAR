"""Architecture from src/model.py, using maintained Keras GroupNormalization."""

import tensorflow as tf


def build_encoder(dropout=0.1, l2=0.0001):
    inputs = tf.keras.Input((98, 64, 1), name="log_mel")
    x = inputs
    for i, width in enumerate((24, 32, 64, 128)):
        branches = []
        for kernel in ((1, 4), (4, 1)):
            y = tf.keras.layers.Conv2D(
                width,
                kernel,
                padding="same",
                kernel_regularizer=tf.keras.regularizers.L2(l2),
            )(x)
            y = tf.keras.layers.GroupNormalization(groups=4, epsilon=1e-3)(y)
            branches.append(tf.keras.layers.ReLU()(y))
        x = tf.keras.layers.Concatenate()(branches)
        x = tf.keras.layers.Conv2D(
            width, 1, kernel_regularizer=tf.keras.regularizers.L2(l2)
        )(x)
        x = tf.keras.layers.GroupNormalization(groups=4, epsilon=1e-3)(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    return tf.keras.Model(
        inputs, tf.keras.layers.GlobalMaxPooling2D()(x), name="encoder"
    )


def build_model(num_classes, dropout=0.1):
    if num_classes < 2:
        raise ValueError("At least two classes required")
    encoder = build_encoder(dropout)
    inputs = tf.keras.Input((98, 64, 1), name="audio_input")
    return tf.keras.Model(
        inputs,
        tf.keras.layers.Dense(num_classes, name="classifier")(encoder(inputs)),
        name="fedstar",
    )


def contrastive_loss(first, second, bilinear):
    """Symmetric cross-view InfoNCE with learned bilinear similarities.

    This explicit choice reconstructs the paper's underspecified bilinear loss;
    it is not a recovered original pretraining implementation.
    """
    logits = tf.matmul(tf.matmul(first, bilinear), second, transpose_b=True)
    targets = tf.range(tf.shape(first)[0])
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits
    return 0.5 * (
        tf.reduce_mean(ce(labels=targets, logits=logits))
        + tf.reduce_mean(ce(labels=targets, logits=tf.transpose(logits)))
    )
