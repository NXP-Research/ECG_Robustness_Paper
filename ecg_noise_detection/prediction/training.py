# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import gc

import config
import numpy as np
import tensorflow as tf
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


@tf.function(reduce_retracing=True)
def process_data_random_masking(
    model, dataset, number_of_batches, num_repetitions, mask_ratio, training=False
):
    distances = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)
    labels = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)

    i = tf.constant(0)

    for batch_data, batch_labels in dataset:
        batch_size = tf.shape(batch_data)[0]
        time_steps = tf.shape(batch_data)[1]

        batch_distances = tf.zeros(batch_size, dtype=tf.float32)

        for _ in tf.range(num_repetitions):
            num_masks = tf.cast(
                tf.math.ceil(mask_ratio * tf.cast(time_steps, tf.float32)), tf.int32
            )

            mask = tf.ones([batch_size, time_steps, 1])

            random_values = tf.random.uniform([batch_size, time_steps])
            _, mask_indices = tf.math.top_k(random_values, k=num_masks)

            batch_indices = tf.repeat(tf.range(batch_size), repeats=num_masks)

            indices = tf.stack(
                [batch_indices, tf.reshape(mask_indices, [-1])],
                axis=1,
            )

            updates_shape = tf.stack([batch_size * num_masks, 1])
            updates = tf.zeros(updates_shape, dtype=tf.float32)

            mask = tf.tensor_scatter_nd_update(mask, indices, updates)

            noise = tf.random.normal(shape=tf.shape(batch_data), mean=0, stddev=1)
            masked_input = batch_data * mask + noise * (1 - mask)

            output = model(masked_input, training=training)
            binary_mask = 1.0 - mask
            masked_original = batch_data * binary_mask
            masked_output = output * binary_mask
            loss = tf.reduce_sum(
                tf.math.square(tf.math.subtract(masked_output, masked_original)),
                axis=[1, 2],
            )
            batch_distances += loss

        distances = distances.write(i, batch_distances)
        labels = labels.write(i, tf.cast(batch_labels, tf.float32))
        i += 1

    return distances.concat(), labels.concat()


@tf.function(reduce_retracing=True)
def process_data_sequential_masking(
    model, dataset, number_of_batches, sequence_length, stride=1, training=False
):
    distances = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)
    labels = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)

    i = tf.constant(0)

    for batch_data, batch_labels in dataset:
        batch_size = tf.shape(batch_data)[0]
        time_steps = tf.shape(batch_data)[1]

        batch_distances = tf.zeros(batch_size, dtype=tf.float32)

        num_mask_positions = tf.maximum(1, (time_steps - sequence_length) // stride + 1)

        for mask_start_idx in tf.range(0, num_mask_positions * stride, delta=stride):
            mask_start = tf.minimum(mask_start_idx, time_steps - sequence_length)
            mask_end = mask_start + sequence_length

            mask = tf.ones([batch_size, time_steps, 1], dtype=tf.float32)

            batch_indices = tf.repeat(tf.range(batch_size), sequence_length)

            time_range = tf.range(mask_start, mask_end)
            time_indices = tf.tile(time_range, [batch_size])

            indices = tf.stack([batch_indices, time_indices], axis=1)

            updates_shape = tf.stack([batch_size * sequence_length, 1])
            updates = tf.zeros(updates_shape, dtype=tf.float32)

            mask = tf.tensor_scatter_nd_update(mask, indices, updates)

            noise = tf.random.normal(
                shape=tf.shape(batch_data), mean=0, stddev=1, dtype=tf.float32
            )
            masked_input = batch_data * mask + noise * (1.0 - mask)

            output = model(masked_input, training=training)

            binary_mask = 1.0 - mask
            masked_original = batch_data * binary_mask
            masked_output = output * binary_mask

            loss = tf.reduce_sum(
                tf.math.square(tf.math.subtract(masked_output, masked_original)),
                axis=[1, 2],
            )
            batch_distances += loss

        distances = distances.write(i, batch_distances)
        labels = labels.write(i, tf.cast(batch_labels, tf.float32))
        i += 1

    return distances.concat(), labels.concat()


@tf.function(reduce_retracing=True)
def process_data_autoregressive_prediction(
    model,
    dataset,
    number_of_batches,
    context_window_size,
    prediction_window_size,
    stride=1,
    training=False,
):
    distances = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)
    labels = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)

    i = tf.constant(0)

    for batch_data, batch_labels in dataset:
        batch_size = tf.shape(batch_data)[0]
        time_steps = tf.shape(batch_data)[1]

        batch_distances = tf.zeros(batch_size, dtype=tf.float32)

        total_window_size = context_window_size + prediction_window_size
        num_window_positions = tf.maximum(
            1, (time_steps - total_window_size + 1) // stride
        )

        for window_idx in tf.range(0, num_window_positions * stride, delta=stride):
            window_start = tf.minimum(window_idx, time_steps - total_window_size)

            context_end = window_start + context_window_size

            context_data = batch_data[:, window_start:context_end, :]

            current_sequence = context_data
            prediction_error = tf.zeros(batch_size, dtype=tf.float32)

            for pred_step in tf.range(prediction_window_size):
                output = model(current_sequence, training=training)

                next_step_pred = output[:, -1, :]

                actual_time_idx = context_end + pred_step
                actual_next_step = batch_data[:, actual_time_idx, :]

                step_error = tf.reduce_sum(
                    tf.square(next_step_pred - actual_next_step), axis=1
                )
                prediction_error += step_error

                current_sequence = tf.concat(
                    [
                        current_sequence[:, 1:, :],
                        tf.expand_dims(next_step_pred, axis=1),
                    ],
                    axis=1,
                )

            batch_distances += prediction_error

        distances = distances.write(i, batch_distances)
        labels = labels.write(i, tf.cast(batch_labels, tf.float32))
        i += 1

    return distances.concat(), labels.concat()


@tf.function(reduce_retracing=True)
def process_data_next_step_prediction(
    model, dataset, number_of_batches, training=False
):
    distances = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)
    labels = tf.TensorArray(tf.float32, size=number_of_batches, dynamic_size=False)

    i = tf.constant(0)

    for batch_data, batch_labels in dataset:
        output = model(batch_data, training=training)

        predictions = output[:, :-1, :]
        targets = batch_data[:, 1:, :]

        batch_distances = tf.reduce_sum(tf.square(predictions - targets), axis=[1, 2])

        distances = distances.write(i, batch_distances)
        labels = labels.write(i, tf.cast(batch_labels, tf.float32))
        i += 1

    return distances.concat(), labels.concat()


def evaluate_next_step_prediction(model, dataset, training=False):
    distances, labels = process_data_next_step_prediction(
        model,
        dataset,
        len(dataset),
        training=training,
    )
    return roc_auc_score(labels.numpy(), distances.numpy())


def evaluate_autoregressive_prediction(model, dataset, training=False):
    distances, labels = process_data_autoregressive_prediction(
        model,
        dataset,
        len(dataset),
        context_window_size=tf.constant(config.CONTEXT_WINDOW_SIZE, dtype=tf.float32),
        prediction_window_size=tf.constant(
            config.PREDICTION_WINDOW_SIZE, dtype=tf.float32
        ),
        stride=tf.constant(config.PREDICTION_WINDOW_STRIDE, dtype=tf.float32),
        training=training,
    )
    return roc_auc_score(labels.numpy(), distances.numpy())


def evaluate_random_masking(model, dataset, training=False):
    distances, labels = process_data_random_masking(
        model,
        dataset,
        len(dataset),
        num_repetitions=tf.constant(config.MASK_REPETITIONS, dtype=tf.int32),
        mask_ratio=tf.constant(config.MASK_RATIO_TEST, dtype=tf.float32),
        training=training,
    )
    return roc_auc_score(labels.numpy(), distances.numpy())


def evaluate_sequential_masking(model, dataset, training=False):
    distances, labels = process_data_sequential_masking(
        model,
        dataset,
        len(dataset),
        sequence_length=tf.constant(config.SEQUENCE_LENGTH_TEST, dtype=tf.int32),
        stride=tf.constant(config.SEQUENCE_STRIDE, dtype=tf.int32),
        training=training,
    )
    return roc_auc_score(labels.numpy(), distances.numpy())


@tf.function(reduce_retracing=True)
def training_next_step_prediction(model, optimizer, train_dataset, training=True):
    train_loss = 0.0

    for batch_data in train_dataset:
        with tf.GradientTape() as tape:
            output = model(batch_data, training=training)

            predictions = output[:, :-1, :]
            targets = batch_data[:, 1:, :]

            primary_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(predictions - targets), axis=[1, 2])
            )
            if model.losses:
                regularization_loss = tf.add_n(model.losses)
                total_loss = primary_loss + regularization_loss
            else:
                total_loss = primary_loss
        train_loss += total_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_avg = train_loss / tf.cast(len(train_dataset), tf.float32)
    return train_loss_avg


@tf.function(reduce_retracing=True)
def validation_next_step_prediction(model, val_dataset, training=False):
    val_loss = 0.0

    for batch_data in val_dataset:
        output = model(batch_data, training=training)

        predictions = output[:, :-1, :]
        targets = batch_data[:, 1:, :]

        loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(predictions - targets), axis=[1, 2])
        )

        val_loss += loss

    val_loss_avg = val_loss / tf.cast(len(val_dataset), tf.float32)
    return val_loss_avg


@tf.function(reduce_retracing=True)
def training_sequential_masking(
    model, optimizer, train_dataset, sequence_length, training=True
):
    train_loss = 0.0

    for batch_data in train_dataset:
        batch_size = tf.shape(batch_data)[0]
        time_steps = tf.shape(batch_data)[1]

        max_start_pos = tf.maximum(1, time_steps - sequence_length + 1)
        mask_starts = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=max_start_pos, dtype=tf.int32
        )

        mask = tf.ones([batch_size, time_steps, 1], dtype=tf.float32)

        batch_indices = tf.repeat(tf.range(batch_size), repeats=sequence_length)
        time_indices_offset = tf.tile(tf.range(sequence_length), [batch_size])
        mask_starts_tiled = tf.repeat(mask_starts, repeats=sequence_length)
        time_indices = mask_starts_tiled + time_indices_offset

        indices = tf.stack([batch_indices, time_indices], axis=1)

        updates = tf.zeros([batch_size * sequence_length, 1], dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        noise = tf.random.normal(
            shape=tf.shape(batch_data), mean=0, stddev=1, dtype=tf.float32
        )
        masked_input = batch_data * mask + noise * (1.0 - mask)

        with tf.GradientTape() as tape:
            output = model(masked_input, training=training)
            binary_mask = 1.0 - mask
            masked_original = batch_data * binary_mask
            masked_output = output * binary_mask
            primary_loss = tf.math.reduce_mean(
                tf.reduce_sum(
                    tf.math.square(tf.math.subtract(masked_output, masked_original)),
                    axis=[1, 2],
                ),
                axis=0,
            )
            if model.losses:
                regularization_loss = tf.add_n(model.losses)
                total_loss = primary_loss + regularization_loss
            else:
                total_loss = primary_loss
        train_loss += total_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_avg = train_loss / tf.cast(len(train_dataset), tf.float32)
    return train_loss_avg


@tf.function(reduce_retracing=True)
def validation_sequential_masking(model, val_dataset, sequence_length, training=False):
    val_loss = 0.0

    for batch_data in val_dataset:
        batch_size = tf.shape(batch_data)[0]
        time_steps = tf.shape(batch_data)[1]

        max_start_pos = tf.maximum(1, time_steps - sequence_length + 1)
        mask_starts = tf.random.uniform(
            shape=[batch_size], minval=0, maxval=max_start_pos, dtype=tf.int32
        )

        mask = tf.ones([batch_size, time_steps, 1], dtype=tf.float32)

        batch_indices = tf.repeat(tf.range(batch_size), repeats=sequence_length)
        time_indices_offset = tf.tile(tf.range(sequence_length), [batch_size])
        mask_starts_tiled = tf.repeat(mask_starts, repeats=sequence_length)
        time_indices = mask_starts_tiled + time_indices_offset

        indices = tf.stack([batch_indices, time_indices], axis=1)

        updates = tf.zeros([batch_size * sequence_length, 1], dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        noise = tf.random.normal(
            shape=tf.shape(batch_data), mean=0, stddev=1, dtype=tf.float32
        )
        masked_input = batch_data * mask + noise * (1.0 - mask)

        output = model(masked_input, training=training)
        binary_mask = 1.0 - mask
        masked_original = batch_data * binary_mask
        masked_output = output * binary_mask
        loss = tf.math.reduce_mean(
            tf.reduce_sum(
                tf.math.square(tf.math.subtract(masked_output, masked_original)),
                axis=[1, 2],
            ),
            axis=0,
        )
        val_loss += loss

    val_loss_avg = val_loss / tf.cast(len(val_dataset), tf.float32)
    return val_loss_avg


@tf.function(reduce_retracing=True)
def training_random_masking(model, optimizer, train_dataset, mask_ratio, training=True):
    train_loss = 0.0

    for batch_data in train_dataset:
        batch_size, time_steps = (
            tf.shape(batch_data)[0],
            tf.shape(batch_data)[1],
        )

        num_masks = tf.cast(
            tf.math.ceil(mask_ratio * tf.cast(time_steps, tf.float32)), tf.int32
        )

        mask = tf.ones([batch_size, time_steps, 1])

        random_values = tf.random.uniform([batch_size, time_steps])
        _, mask_indices = tf.math.top_k(random_values, k=num_masks)

        batch_indices = tf.repeat(tf.range(batch_size), repeats=num_masks)

        indices = tf.stack([batch_indices, tf.reshape(mask_indices, [-1])], axis=1)

        updates_shape = tf.stack([batch_size * num_masks, 1])
        updates = tf.zeros(updates_shape, dtype=tf.float32)

        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        noise = tf.random.normal(shape=tf.shape(batch_data), mean=0, stddev=1)
        masked_input = batch_data * mask + noise * (1 - mask)

        with tf.GradientTape() as tape:
            output = model(masked_input, training=training)
            binary_mask = 1.0 - mask
            masked_original = batch_data * binary_mask
            masked_output = output * binary_mask
            primary_loss = tf.math.reduce_mean(
                tf.reduce_sum(
                    tf.math.square(tf.math.subtract(masked_output, masked_original)),
                    axis=[1, 2],
                ),
                axis=0,
            )
            if model.losses:
                regularization_loss = tf.add_n(model.losses)
                total_loss = primary_loss + regularization_loss
            else:
                total_loss = primary_loss
        train_loss += total_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_avg = train_loss / tf.cast(len(train_dataset), tf.float32)
    return train_loss_avg


@tf.function(reduce_retracing=True)
def validation_random_masking(model, val_dataset, mask_ratio, training=False):
    val_loss = 0.0
    for batch_data in val_dataset:
        batch_size, time_steps = (
            tf.shape(batch_data)[0],
            tf.shape(batch_data)[1],
        )

        num_masks = tf.cast(
            tf.math.ceil(mask_ratio * tf.cast(time_steps, tf.float32)), tf.int32
        )

        mask = tf.ones([batch_size, time_steps, 1])

        random_values = tf.random.uniform([batch_size, time_steps])
        _, mask_indices = tf.math.top_k(random_values, k=num_masks)

        batch_indices = tf.repeat(tf.range(batch_size), repeats=num_masks)

        indices = tf.stack([batch_indices, tf.reshape(mask_indices, [-1])], axis=1)

        updates_shape = tf.stack([batch_size * num_masks, 1])
        updates = tf.zeros(updates_shape, dtype=tf.float32)

        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        noise = tf.random.normal(shape=tf.shape(batch_data), mean=0, stddev=1)
        masked_input = batch_data * mask + noise * (1 - mask)

        output = model(masked_input, training=training)
        binary_mask = 1.0 - mask
        masked_original = batch_data * binary_mask
        masked_output = output * binary_mask
        loss = tf.math.reduce_mean(
            tf.reduce_sum(
                tf.math.square(tf.math.subtract(masked_output, masked_original)),
                axis=[1, 2],
            ),
            axis=0,
        )
        val_loss += loss

    val_loss_avg = val_loss / tf.cast(len(val_dataset), tf.float32)
    return val_loss_avg


def evaluate_by_method(model, dataset, method, training=False):
    evaluation_methods = {
        "random_masking": evaluate_random_masking,
        "sequential_masking": evaluate_sequential_masking,
        "autoregressive_prediction": evaluate_autoregressive_prediction,
        "next_step_prediction": evaluate_next_step_prediction,
    }

    if method not in evaluation_methods:
        raise ValueError(f"Unknown testing method: {method}")
    return evaluation_methods[method](model, dataset, training=training)


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    val_auc_dataset,
    num_epochs,
    patience,
    val_frequency,
    break_early,
):
    best_auc = -1
    best_weights = model.get_weights()
    lr_status = 0
    patience_checks = patience // val_frequency

    for epoch in range(num_epochs):
        if config.TRAINING_METHOD == "random_masking":
            train_loss_avg = training_random_masking(
                model,
                optimizer,
                train_dataset,
                mask_ratio=tf.constant(config.MASK_RATIO_TRAIN, dtype=tf.float32),
                training=True,
            ).numpy()

        elif config.TRAINING_METHOD == "sequential_masking":
            train_loss_avg = training_sequential_masking(
                model,
                optimizer,
                train_dataset,
                sequence_length=tf.constant(
                    config.SEQUENCE_LENGTH_TRAIN, dtype=tf.int32
                ),
                training=True,
            ).numpy()

        elif config.TRAINING_METHOD == "next_step_prediction":
            train_loss_avg = training_next_step_prediction(
                model,
                optimizer,
                train_dataset,
                training=True,
            ).numpy()

        if epoch % val_frequency == 0:
            if config.TRAINING_METHOD == "random_masking":
                val_loss_avg = validation_random_masking(
                    model,
                    val_dataset,
                    mask_ratio=tf.constant(config.MASK_RATIO_TRAIN, dtype=tf.float32),
                    training=False,
                ).numpy()

            elif config.TRAINING_METHOD == "sequential_masking":
                val_loss_avg = validation_sequential_masking(
                    model,
                    val_dataset,
                    sequence_length=tf.constant(
                        config.SEQUENCE_LENGTH_TRAIN, dtype=tf.int32
                    ),
                    training=False,
                ).numpy()

            elif config.TRAINING_METHOD == "next_step_prediction":
                val_loss_avg = validation_next_step_prediction(
                    model, val_dataset, training=False
                ).numpy()

            val_auc = evaluate_by_method(
                model, val_auc_dataset, config.TESTING_METHOD, training=False
            )

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


def instance_normalize(data, epsilon=1e-6):
    mean_instances = np.mean(data, axis=1, keepdims=True)
    std_instances = np.std(data, axis=1, keepdims=True)

    std_instances_safe = np.where(std_instances < epsilon, 1.0, std_instances)

    return (data - mean_instances) / std_instances_safe
