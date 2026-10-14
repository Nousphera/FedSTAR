"""FedSTAR Algorithm 1: confidence-filtered self-training and weighted FedAvg."""

import numpy as np
import tensorflow as tf


def confidence_threshold(completed, rounds):
    if rounds < 1 or not 0 <= completed <= rounds:
        raise ValueError("Invalid round")
    return float(0.9 - 0.4 * 0.5 * (1 + np.cos(np.pi * completed / rounds)))


def self_training_loss(
    logits, labels, threshold=0.5, temperature=4.0, beta=0.5, mode="paper"
):
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    labeled = labels >= 0
    llogits = tf.boolean_mask(logits, labeled)
    llabels = tf.boolean_mask(labels, labeled)
    ulogits = tf.boolean_mask(logits, ~labeled)
    probabilities = tf.stop_gradient(tf.nn.softmax(ulogits / temperature))
    mask = tf.cast(tf.reduce_max(probabilities, axis=-1) >= threshold, tf.float32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits
    ls = tf.math.divide_no_nan(
        tf.reduce_sum(ce(labels=llabels, logits=llogits)),
        tf.cast(tf.size(llabels), tf.float32),
    )
    lu = tf.math.divide_no_nan(
        tf.reduce_sum(
            ce(labels=tf.argmax(probabilities, axis=-1), logits=ulogits) * mask
        ),
        tf.cast(tf.shape(ulogits)[0], tf.float32),
    )
    loss = ls + beta * lu if mode == "paper" else (1 - beta) * ls + beta * lu
    return loss, {
        "labeled_loss": ls,
        "unlabeled_loss": lu,
        "accepted": tf.reduce_sum(mask),
        "unlabeled": tf.shape(ulogits)[0],
    }


def aggregate(updates):
    if not updates or any(n <= 0 for _, n in updates):
        raise ValueError("Positive example counts required")
    total = sum(n for _, n in updates)
    reference = updates[0][0]
    if any(
        len(w) != len(reference)
        or any(a.shape != b.shape for a, b in zip(w, reference))
        for w, _ in updates
    ):
        raise ValueError("Mismatched model weights")
    return [
        sum(
            np.asarray(w[i], dtype=np.float64) * (n / total) for w, n in updates
        ).astype(reference[i].dtype)
        for i in range(len(reference))
    ]
