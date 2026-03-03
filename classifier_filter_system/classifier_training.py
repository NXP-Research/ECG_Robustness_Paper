# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import config
import numpy as np
import scipy.signal as signal
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve


def loss_fn_multilabel(model, data, labels, training=False):
    logits = model(data, training=training)
    bce = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction="sum_over_batch_size"
    )
    loss = bce(labels, logits)
    return loss


def get_best_macro_f1_score(y_true, y_pred):
    n_labels = y_true.shape[1]
    optimal_thresholds = []
    best_f1_scores = []

    for label_idx in range(n_labels):
        true_labels = y_true[:, label_idx]
        pred_probs = y_pred[:, label_idx]

        precision, recall, thresholds = precision_recall_curve(true_labels, pred_probs)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

        if thresholds.size == 0:
            optimal_threshold = 0.5
            best_f1 = 0.0
        else:
            best_f1_idx = int(np.argmax(f1_scores[:-1]))
            optimal_threshold = thresholds[best_f1_idx]
            best_f1 = f1_scores[best_f1_idx]
        optimal_thresholds.append(optimal_threshold)
        best_f1_scores.append(best_f1)

    best_macro_f1 = np.mean(best_f1_scores)

    return best_macro_f1, optimal_thresholds


def thresholds_max_subset_acc(model, val_dataset, max_iters=5):
    Xs, Ys = [], []
    for X_b, Y_b in val_dataset:
        Xs.append(X_b.numpy())
        Ys.append(Y_b.numpy().astype(np.int32))
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)

    logits = model(X, training=False)
    probs = tf.nn.sigmoid(logits).numpy()
    L = probs.shape[1]

    # Build per-label candidate thresholds (unique probs or quantiles)
    cand_list = []
    for l in range(L):
        p = probs[:, l]
        uniq = np.unique(p)
        # include tiny margins so the step function is well-defined
        cand = np.r_[uniq.min() - 1e-6, uniq, uniq.max() + 1e-6]
        cand_list.append(cand)

    # Initialize thresholds from (fixed) per-label F1 thresholds with safe indexing
    thr = np.zeros(L, dtype=np.float32)
    for l in range(L):
        precision, recall, thresholds = precision_recall_curve(Y[:, l], probs[:, l])
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        if thresholds.size == 0:
            t0 = 0.5
        else:
            # thresholds aligns with precision[1:] and recall[1:]
            idx = int(np.argmax(f1[:-1])) if f1.size > 1 else 0
            t0 = float(thresholds[idx])
        thr[l] = t0

    def subset_acc_for(thr_vec):
        preds = (probs > thr_vec[None, :]).astype(np.int32)
        return float((preds == Y).all(axis=1).mean())

    best_acc = subset_acc_for(thr)

    # Coordinate ascent
    for _ in range(max_iters):
        improved = False
        for l in range(L):
            best_l = thr[l]
            best_acc_l = best_acc
            for t in cand_list[l]:
                thr_try = thr.copy()
                thr_try[l] = t
                acc_try = subset_acc_for(thr_try)
                if acc_try > best_acc_l + 1e-12:
                    best_acc_l = acc_try
                    best_l = t
            if best_l != thr[l]:
                thr[l] = best_l
                best_acc = best_acc_l
                improved = True
        if not improved:
            break
    return thr, best_acc


def get_best_auc_thresholds(y_true, y_prob):
    best_thresholds = []
    n_labels = y_true.shape[1]

    for label_idx in range(n_labels):
        y_true_label = y_true[:, label_idx]
        y_pred_label_probs = y_prob[:, label_idx]

        fpr, tpr, thresholds = roc_curve(y_true_label, y_pred_label_probs)
        j_statistic = tpr - fpr

        best_threshold_idx = np.argmax(j_statistic)
        best_threshold = thresholds[best_threshold_idx]
        best_thresholds.append(best_threshold)

    return best_thresholds


def evaluate_auc(model, test_dataset, training=False):
    probs_array = []
    label_array = []
    for batch_data, batch_labels in test_dataset:
        logits = model(batch_data, training=training)
        probs = tf.nn.sigmoid(logits)
        probs_array.append(probs.numpy())
        label_array.append(batch_labels.numpy())
    probs_array = np.concatenate(probs_array)
    label_array = np.concatenate(label_array)
    macro_roc_auc_ovr = roc_auc_score(
        y_true=label_array, y_score=probs_array, average="macro"
    )
    return macro_roc_auc_ovr


def evaluate_f1(model, test_dataset, training=False):
    probs_array = []
    label_array = []
    for batch_data, batch_labels in test_dataset:
        logits = model(batch_data, training=training)
        probs = tf.nn.sigmoid(logits)
        probs_array.append(probs.numpy())
        label_array.append(batch_labels.numpy())
    probs_array = np.concatenate(probs_array)
    label_array = np.concatenate(label_array)
    f1, _ = get_best_macro_f1_score(y_true=label_array, y_pred=probs_array)
    return f1


def evaluate_acc(
    model, test_dataset, test_set_size, filter_set_size, training=False, thresholds=None
):
    probs_array = []
    label_array = []
    for batch_data, batch_labels in test_dataset:
        logits = model(batch_data, training=training)
        probs = tf.nn.sigmoid(logits)
        probs_array.append(probs.numpy())
        label_array.append(batch_labels.numpy())
    probs_array = np.concatenate(probs_array)
    label_array = np.concatenate(label_array)
    if thresholds is None:
        thresholds, _ = thresholds_max_subset_acc(model, test_dataset)
    one_hot = (probs_array > np.asarray(thresholds)).astype(np.int32)
    correct = int((one_hot == label_array).all(axis=1).sum())
    subset_accuracy_without_filter_data = float(correct) / float(test_set_size)
    subset_accuracy_with_filter_data = float(correct) / float(
        test_set_size + filter_set_size
    )
    return (
        subset_accuracy_without_filter_data,
        subset_accuracy_with_filter_data,
        thresholds,
    )


def train(model, train_dataset, val_dataset, num_epochs, learning_rate, patience):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    best_auc = -1
    best_weights = model.get_weights()
    lr_status = 0
    patience_counter = patience
    for epoch in range(num_epochs):
        train_loss = 0

        for batch_data, batch_labels in train_dataset:
            with tf.GradientTape() as tape:
                loss = loss_fn_multilabel(
                    model, batch_data, batch_labels, training=True
                )
            train_loss += loss.numpy()

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss_avg = train_loss / len(train_dataset)

        val_loss = 0
        for batch_data, batch_labels in val_dataset:
            loss = loss_fn_multilabel(model, batch_data, batch_labels, training=False)
            val_loss += loss.numpy()

        val_loss_avg = val_loss / len(val_dataset)

        auc = evaluate_auc(model, val_dataset, training=False)

        if auc > best_auc:
            best_auc = auc
            best_weights = model.get_weights()
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0 and lr_status == 0:
                print("PLATEAU 1!")
                optimizer.learning_rate.assign(learning_rate * 0.1)
                patience_counter = patience
                lr_status = 1
                model.set_weights(best_weights)
            if patience_counter == 0 and lr_status == 1:
                print("PLATEAU 2!")
                optimizer.learning_rate.assign(learning_rate * 0.01)
                patience_counter = patience
                lr_status = 2
                model.set_weights(best_weights)
            if patience_counter == 0 and lr_status == 2:
                print("PLATEAU 3!")
                break

        print(
            "epoch:",
            epoch,
            "train_loss:",
            train_loss_avg,
            "val_loss:",
            val_loss_avg,
            "auc:",
            auc,
        )

    model.set_weights(best_weights)


def main():
    CONDITION = "MI"

    x_train = np.load(f"data/ptb_xl/x_train_{CONDITION}.npy")
    x_train = signal.resample(x=x_train, num=config.TIME_STEPS, axis=1).astype(
        np.float32
    )
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std

    x_val = np.load(f"data/ptb_xl/x_val_non_ood_clean_{CONDITION}.npy")
    x_val = signal.resample(x=x_val, num=config.TIME_STEPS, axis=1).astype(np.float32)
    x_val = (x_val - train_mean) / train_std

    x_test = np.load(f"data/ptb_xl/x_test_non_ood_clean_{CONDITION}.npy")
    x_test = signal.resample(x=x_test, num=config.TIME_STEPS, axis=1).astype(np.float32)
    x_test = (x_test - train_mean) / train_std

    y_train = np.load(f"data/ptb_xl/y_train_{CONDITION}.npy").astype(np.int32)
    y_val = np.load(f"data/ptb_xl/y_val_non_ood_clean_{CONDITION}.npy").astype(np.int32)
    y_test = np.load(f"data/ptb_xl/y_test_non_ood_clean_{CONDITION}.npy").astype(
        np.int32
    )

    filterdata_val1 = np.load(f"data/ptb_xl/x_val_non_ood_noisy_{CONDITION}.npy")
    filterdata_val2 = np.load(f"data/ptb_xl/x_val_ood_clean_{CONDITION}.npy")
    filterdata_val3 = np.load(f"data/ptb_xl/x_val_ood_noisy_{CONDITION}.npy")

    filterdata_test1 = np.load(f"data/ptb_xl/x_test_non_ood_noisy_{CONDITION}.npy")
    filterdata_test2 = np.load(f"data/ptb_xl/x_test_ood_clean_{CONDITION}.npy")
    filterdata_test3 = np.load(f"data/ptb_xl/x_test_ood_noisy_{CONDITION}.npy")

    filter_set_size_val = (
        filterdata_val1.shape[0] + filterdata_val2.shape[0] + filterdata_val3.shape[0]
    )
    filter_set_size_test = (
        filterdata_test1.shape[0]
        + filterdata_test2.shape[0]
        + filterdata_test3.shape[0]
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(x_train.shape[0], reshuffle_each_iteration=True)
        .batch(256, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(x_val.shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(x_test.shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )

    model = ...  # put resnet1d_wang model here

    auc = evaluate_auc(model, test_dataset, training=False)
    _, _, best_thresholds = evaluate_acc(
        model, val_dataset, x_val.shape[0], filter_set_size_val, training=False
    )
    acc_clean, acc_with_filter_data, _ = evaluate_acc(
        model,
        test_dataset,
        x_test.shape[0],
        filter_set_size_test,
        training=False,
        thresholds=best_thresholds,
    )
    print("Initial AUC:", auc, "Initial ACC:", acc_clean, acc_with_filter_data)

    train(
        model,
        train_dataset,
        val_dataset,
        num_epochs=100,
        learning_rate=0.01,
        patience=20,
    )

    auc = evaluate_auc(model, test_dataset, training=False)
    _, _, best_thresholds = evaluate_acc(
        model, val_dataset, x_val.shape[0], filter_set_size_val, training=False
    )
    acc_clean, acc_with_filter_data, _ = evaluate_acc(
        model,
        test_dataset,
        x_test.shape[0],
        filter_set_size_test,
        training=False,
        thresholds=best_thresholds,
    )
    print(
        "FINAL AUC:",
        auc,
        "FINAL ACC (without filter data):",
        acc_clean,
        "FINAL CUSTOM ACC (with filter data):",
        acc_with_filter_data,
    )

    model.save(
        f"classifier_filter_system/classifier_model_{CONDITION}.keras"
    )
    np.save(
        f"classifier_filter_system/classifier_thresholds_{CONDITION}.npy",
        np.asarray(best_thresholds, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
