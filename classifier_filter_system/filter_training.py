# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import gc
import math
import os

import numpy as np
import tensorflow as tf
from scipy import signal
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import config
from filter_model import ResNet

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

label_positions = {"HYP": 1, "CD": 2, "STTC": 3, "MI": 4}


def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()

    if gpus:
        try:
            for gpu in gpus:
                device_name = ":".join(gpu.name.split(":")[-2:])
                tf.config.experimental.reset_memory_stats(device_name)
        except Exception as e:
            print(f"Warning: Could not reset memory stats: {e}")


def get_center(model, dataset, data_count, latent_dim):
    avg = np.zeros((latent_dim))
    for batch_data in dataset:
        output = model(batch_data, training=False)
        avg += np.sum(output.numpy(), axis=0)
    return avg / data_count


def get_best_macro_f1_score(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

    best_f1_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]

    return best_f1, optimal_threshold


def get_best_auc_thresholds(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    j_statistic = tpr - fpr

    best_threshold_idx = np.argmax(j_statistic)
    best_threshold = thresholds[best_threshold_idx]

    return best_threshold


def evaluate_auc(model, dataset, center, training=False):
    l2_array = []
    label_array = []
    for batch_data, batch_labels, _ in dataset:
        output = model(batch_data, training=training)
        dist = tf.reduce_sum(tf.math.square(tf.math.subtract(output, center)), axis=1)
        l2_array.append(dist.numpy())
        label_array.append(batch_labels.numpy())
    l2_array = np.concatenate(l2_array)
    label_array = np.concatenate(label_array)
    auc = roc_auc_score(
        label_array,
        l2_array,
    )
    return auc


def evaluate_acc2(
    model,
    classifier_model,
    classifier_thresholds,
    dataset,
    center,
    training=False,
):

    # 1) Collect all batches
    X_list, y_bin_list, y_multi_list, d_list = [], [], [], []
    for x_b, y_b_binary, y_b_multi in dataset:
        z_b = model(x_b, training=training)
        d_b = tf.reduce_sum(tf.square(z_b - center), axis=1).numpy()
        X_list.append(x_b.numpy())
        y_bin_list.append(y_b_binary.numpy().astype(np.int32))
        y_multi_list.append(y_b_multi.numpy().astype(np.int32))
        d_list.append(d_b)

    X = np.concatenate(X_list, axis=0)
    y_bin = np.concatenate(y_bin_list, axis=0)
    y_multi = np.concatenate(y_multi_list, axis=0)
    dists = np.concatenate(d_list, axis=0)

    # 2) Classifier forward pass once
    probs = tf.nn.sigmoid(classifier_model(X, training=False)).numpy()
    preds = (probs > np.asarray(classifier_thresholds)).astype(np.int32)

    # Subset correctness for accepted samples
    subset_correct = (preds == y_multi).all(axis=1).astype(np.int32)

    # 3) Order by distance ascending (threshold t = d[i] accepts prefix 0..i)
    order = np.argsort(dists)
    rej_order = (y_bin[order] == 1).astype(np.int32)  # should reject
    acc_corr_order = ((y_bin[order] == 0) & (subset_correct[order] == 1)).astype(
        np.int32
    )

    # 4) Prefix/suffix cumulative sums
    cum_acc_corr = np.cumsum(acc_corr_order).astype(
        np.int64
    )  # accepted & correct in prefix
    cum_rej = np.cumsum(rej_order).astype(np.int64)  # rejects inside prefix
    total_rej = int(rej_order.sum())
    correct_rejects_suffix = total_rej - cum_rej  # correctly rejected in suffix (> t)

    # 5) Accuracy and reject rate for every t = d_sorted[i]
    total_correct = cum_acc_corr + correct_rejects_suffix
    N = len(order)
    accs = total_correct / float(N)
    rej_rates = (N - (np.arange(N) + 1)) / float(N)  # fraction rejected

    best_idx = int(np.argmax(accs))
    best_acc = float(accs[best_idx])

    return best_acc, accs, rej_rates


def custom_acc_classifier_only(classifier_model, thresholds, dataset):
    X_all, y_bin_all, y_multi_all = [], [], []
    for x_b, y_b_bin, y_b_multi in dataset:
        X_all.append(x_b.numpy())
        y_bin_all.append(y_b_bin.numpy().astype(np.int32))
        y_multi_all.append(y_b_multi.numpy().astype(np.int32))
    X = np.concatenate(X_all, axis=0)
    y_bin = np.concatenate(y_bin_all, axis=0)
    y_multi = np.concatenate(y_multi_all, axis=0)

    probs = tf.nn.sigmoid(classifier_model(X, training=False)).numpy()
    preds = (probs > np.asarray(thresholds)).astype(np.int32)

    correct_cls = np.all(preds == y_multi, axis=1).astype(np.int32)
    custom_correct = (y_bin == 0) * correct_cls
    return float(custom_correct.mean())


@tf.function
def training(model, optimizer, train_dataset, center, training=True):
    train_loss = 0.0
    for batch_data in train_dataset:
        with tf.GradientTape() as tape:
            output = model(batch_data, training=training)
            loss = tf.reduce_mean(
                tf.reduce_sum(tf.math.square(tf.math.subtract(output, center)), axis=1),
                axis=0,
            )
        train_loss += loss

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_avg = train_loss / tf.cast(len(train_dataset), tf.float32)
    return train_loss_avg


@tf.function
def validation(model, val_dataset, center, training=False):
    val_loss = 0.0
    for batch_data in val_dataset:
        output = model(batch_data, training=training)
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.math.square(tf.math.subtract(output, center)), axis=1),
            axis=0,
        )
        val_loss += loss

    val_loss_avg = val_loss / tf.cast(len(val_dataset), tf.float32)
    return val_loss_avg


def train(
    model,
    classifier_model,
    classifier_thresholds,
    optimizer,
    train_dataset,
    val_auc_dataset,
    num_epochs,
    patience,
    val_frequency,
    break_early,
    center,
):
    best_auc = -1
    best_weights = model.get_weights()
    lr_status = 0
    patience_checks = max(1, math.ceil(patience / val_frequency))

    for epoch in range(num_epochs):
        train_loss_avg = training(
            model, optimizer, train_dataset, center, training=True
        ).numpy()

        if epoch % val_frequency == 0:
            val_auc = evaluate_auc(model, val_auc_dataset, center, training=False)
            val_acc, _, _ = evaluate_acc2(
                model,
                classifier_model,
                classifier_thresholds,
                val_auc_dataset,
                center,
                training=False,
            )

            print(
                "epoch:",
                epoch,
                "train_loss:",
                train_loss_avg,
                "val_auc:",
                val_auc,
                "val_acc:",
                val_acc,
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


def get_model_from_trial_params(params):
    num_layers = params["num_layers"]
    num_filters_list = [params[f"filters_{i}"] for i in range(num_layers)]
    kernel_size_list = [params[f"kernel_{i}"] for i in range(num_layers)]
    strides_list = [params[f"stride_{i}"] for i in range(num_layers)]

    model = ResNet(
        initial_num_filter=params["initial_num_filter"],
        initial_kernel_size=params["initial_kernel_size"],
        initial_stride=params["initial_stride"],
        num_filters_list=num_filters_list,
        kernel_size_list=kernel_size_list,
        strides_list=strides_list,
        latent_dim=params["latent_dim"],
    )
    return model


def main():
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    CONDITION = "MI"

    x_train = np.load(f"data/ptb_xl/x_train_{CONDITION}.npy")
    x_train = signal.resample(x=x_train, num=config.TIME_STEPS, axis=1).astype(
        np.float32
    )
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std

    x_val_auc_ood_noisy = np.load(f"data/ptb_xl/x_val_ood_noisy_{CONDITION}.npy")
    x_val_auc_ood_clean = np.load(f"data/ptb_xl/x_val_ood_clean_{CONDITION}.npy")
    x_val_auc_non_ood_noisy = np.load(
        f"data/ptb_xl/x_val_non_ood_noisy_{CONDITION}.npy"
    )
    x_val_auc_non_ood_clean = np.load(
        f"data/ptb_xl/x_val_non_ood_clean_{CONDITION}.npy"
    )

    x_val_auc = np.concatenate(
        [
            x_val_auc_ood_noisy,
            x_val_auc_ood_clean,
            x_val_auc_non_ood_noisy,
            x_val_auc_non_ood_clean,
        ]
    )
    x_val_auc = signal.resample(x=x_val_auc, num=config.TIME_STEPS, axis=1).astype(
        np.float32
    )
    x_val_auc = (x_val_auc - train_mean) / train_std

    x_test_auc_ood_noisy = np.load(f"data/ptb_xl/x_test_ood_noisy_{CONDITION}.npy")
    x_test_auc_ood_clean = np.load(f"data/ptb_xl/x_test_ood_clean_{CONDITION}.npy")
    x_test_auc_non_ood_noisy = np.load(
        f"data/ptb_xl/x_test_non_ood_noisy_{CONDITION}.npy"
    )
    x_test_auc_non_ood_clean = np.load(
        f"data/ptb_xl/x_test_non_ood_clean_{CONDITION}.npy"
    )

    x_test = np.concatenate(
        [
            x_test_auc_ood_noisy,
            x_test_auc_ood_clean,
            x_test_auc_non_ood_noisy,
            x_test_auc_non_ood_clean,
        ]
    )
    x_test = signal.resample(x=x_test, num=config.TIME_STEPS, axis=1).astype(np.float32)
    x_test = (x_test - train_mean) / train_std

    y_val_non_ood_clean = np.load(
        f"data/ptb_xl/y_val_non_ood_clean_{CONDITION}.npy"
    ).astype(np.int32)
    y_test_non_ood_clean = np.load(
        f"data/ptb_xl/y_test_non_ood_clean_{CONDITION}.npy"
    ).astype(np.int32)

    L_val = y_val_non_ood_clean.shape[1]
    L_test = y_test_non_ood_clean.shape[1]
    assert L_val == L_test, "Val/Test multilabel dimensionality mismatch."
    L_all = L_val

    n_val_on = x_val_auc_ood_noisy.shape[0]
    n_val_oc = x_val_auc_ood_clean.shape[0]
    n_val_nn = x_val_auc_non_ood_noisy.shape[0]
    n_val_nc = x_val_auc_non_ood_clean.shape[0]

    n_test_on = x_test_auc_ood_noisy.shape[0]
    n_test_oc = x_test_auc_ood_clean.shape[0]
    n_test_nn = x_test_auc_non_ood_noisy.shape[0]
    n_test_nc = x_test_auc_non_ood_clean.shape[0]

    y_alllabels_val = np.concatenate(
        [
            np.zeros((n_val_on, L_all), dtype=np.int32),
            np.zeros((n_val_oc, L_all), dtype=np.int32),
            np.zeros((n_val_nn, L_all), dtype=np.int32),
            y_val_non_ood_clean,
        ],
        axis=0,
    )

    y_alllabels_test = np.concatenate(
        [
            np.zeros((n_test_on, L_all), dtype=np.int32),
            np.zeros((n_test_oc, L_all), dtype=np.int32),
            np.zeros((n_test_nn, L_all), dtype=np.int32),
            y_test_non_ood_clean,
        ],
        axis=0,
    )

    y_val_bin = np.concatenate(
        [
            np.ones(n_val_on, dtype=np.int32),
            np.ones(n_val_oc, dtype=np.int32),
            np.ones(n_val_nn, dtype=np.int32),
            np.zeros(n_val_nc, dtype=np.int32),
        ],
        axis=0,
    )

    y_test_bin = np.concatenate(
        [
            np.ones(n_test_on, dtype=np.int32),
            np.ones(n_test_oc, dtype=np.int32),
            np.ones(n_test_nn, dtype=np.int32),
            np.zeros(n_test_nc, dtype=np.int32),
        ],
        axis=0,
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(x_train.shape[0], reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_auc_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val_auc, y_val_bin, y_alllabels_val))
        .batch(x_val_auc.shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_bin, y_alllabels_test))
        .batch(x_test.shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )

    specific_params = {
        "num_layers": 2,
        "initial_num_filter": 12,
        "initial_kernel_size": 11,
        "initial_stride": 1,
        "filters_0": 24,
        "kernel_0": 9,
        "stride_0": 1,
        "filters_1": 128,
        "kernel_1": 3,
        "stride_1": 1,
        "latent_dim": 128,
    }
    model = get_model_from_trial_params(specific_params)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE)
    center = tf.convert_to_tensor(
        get_center(model, train_dataset, x_train.shape[0], config.LATENT_DIM),
        dtype=tf.float32,
    )

    classifier_model = tf.keras.models.load_model(
        f"classifier_filter_system/classifier_model_{CONDITION}.keras"
    )

    classifier_thresholds = np.load(
        f"classifier_filter_system/classifier_thresholds_{CONDITION}.npy"
    )

    baseline = custom_acc_classifier_only(
        classifier_model, classifier_thresholds, test_dataset
    )

    print("Baseline classifier:", baseline)

    test_auc = evaluate_auc(model, test_dataset, center, training=False)
    test_acc, _, _ = evaluate_acc2(
        model,
        classifier_model,
        classifier_thresholds,
        test_dataset,
        center,
        training=False,
    )
    print("Initial AUC:", test_auc, "Initial ACC:", test_acc)

    train(
        model,
        classifier_model,
        classifier_thresholds,
        optimizer,
        train_dataset,
        val_auc_dataset,
        num_epochs=config.RETRAIN_EPOCHS,
        patience=config.RETRAIN_PATIENCE,
        val_frequency=config.RETRAIN_VAL_FREQUENCY,
        break_early=False,
        center=center,
    )

    test_auc = evaluate_auc(model, test_dataset, center, training=False)
    test_acc, accs, rejs = evaluate_acc2(
        model,
        classifier_model,
        classifier_thresholds,
        test_dataset,
        center,
        training=False,
    )
    np.save(f"classifier_filter_system/accs_{CONDITION}.npy", accs)
    np.save(f"classifier_filter_system/rejs_{CONDITION}.npy", rejs)
    print("Final AUC Filter:", test_auc, "Final CUSTOM ACC Classifier+Filter System:", test_acc)


if __name__ == "__main__":
    main()
