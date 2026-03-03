# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import gc

import config
import numpy as np
import tensorflow as tf
from model import VariationalAutoencoder
from sklearn.metrics import roc_auc_score


def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting up GPU memory growth: {e}")


def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()


def evaluate(model, dataset, training=False):
    l2_array = []
    label_array = []
    for batch_data, batch_labels in dataset:
        output = model(batch_data, training=training)

        reconstruction_distances = tf.reduce_sum(
            tf.math.square(tf.math.subtract(output, batch_data)), axis=[1, 2]
        )

        score = reconstruction_distances

        l2_array.append(score.numpy())
        label_array.append(batch_labels.numpy())
    l2_array = np.concatenate(l2_array)
    label_array = np.concatenate(label_array)
    auc = roc_auc_score(
        label_array,
        l2_array,
    )
    return auc


@tf.function
def training(model, optimizer, train_dataset, kl_loss_beta, training=True):
    train_loss = 0.0

    for batch_data in train_dataset:
        with tf.GradientTape() as tape:
            output = model(batch_data, training=training)

            reconstruction_loss = tf.math.reduce_mean(
                tf.reduce_sum(
                    tf.math.square(tf.math.subtract(output, batch_data)), axis=[1, 2]
                ),
                axis=0,
            )

            loss = reconstruction_loss

            if isinstance(model, VariationalAutoencoder):
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(
                        1
                        + model.z_log_var
                        - tf.square(model.z_mean)
                        - tf.exp(model.z_log_var),
                        axis=1,
                    )
                )
                loss += kl_loss_beta * kl_loss

        train_loss += loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_avg = train_loss / tf.cast(len(train_dataset), tf.float32)
    return train_loss_avg


@tf.function
def validation(model, val_dataset, kl_loss_beta, training=False):
    val_loss = 0.0
    for batch_data in val_dataset:
        output = model(batch_data, training=training)

        reconstruction_loss = tf.math.reduce_mean(
            tf.reduce_sum(
                tf.math.square(tf.math.subtract(output, batch_data)), axis=[1, 2]
            ),
            axis=0,
        )

        loss = reconstruction_loss

        if isinstance(model, VariationalAutoencoder):
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + model.z_log_var
                    - tf.square(model.z_mean)
                    - tf.exp(model.z_log_var),
                    axis=1,
                )
            )
            loss += kl_loss_beta * kl_loss

        val_loss += loss

    val_loss_avg = val_loss / tf.cast(len(val_dataset), tf.float32)
    return val_loss_avg


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    val_auc_dataset,
    num_epochs,
    patience,
    val_frequency,
    kl_loss_beta,
    break_early,
):
    best_auc = -1
    best_weights = model.get_weights()
    lr_status = 0
    patience_checks = patience // val_frequency

    for epoch in range(num_epochs):
        train_loss_avg = training(
            model, optimizer, train_dataset, kl_loss_beta, training=True
        ).numpy()

        if epoch % val_frequency == 0:
            val_loss_avg = validation(
                model, val_dataset, kl_loss_beta, training=False
            ).numpy()
            val_auc = evaluate(model, val_auc_dataset, training=False)

            print(
                "epoch:",
                epoch,
                "train_loss:",
                train_loss_avg,
                "val_loss:",
                val_loss_avg,
                "val_auc:",
                val_auc,
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_weights = model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_checks:
                    if lr_status == 0:
                        print("PLATEAU 1!")
                        optimizer.learning_rate.assign(config.LEARNING_RATE_PLATEAU_1)
                        patience_counter = 0
                        lr_status = 1
                        model.set_weights(best_weights)
                        if break_early:
                            break
                    elif lr_status == 1:
                        print("PLATEAU 2!")
                        optimizer.learning_rate.assign(config.LEARNING_RATE_PLATEAU_2)
                        patience_counter = 0
                        lr_status = 2
                        model.set_weights(best_weights)
                    elif lr_status == 2:
                        print("PLATEAU 3!")
                        break

    model.set_weights(best_weights)
    return best_auc
