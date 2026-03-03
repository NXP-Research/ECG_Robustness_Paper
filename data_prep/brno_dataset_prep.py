# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

data_path = Path(...)  # path to the brno university ecg quality dataset folder that contains the record folders 100001 to 126001"

recording_names = [
    "100001",
    "100002",
    "103001",
    "103002",
    "103003",
    "104001",
    "105001",
    "115001",
    "118001",
    "121001",
    "126001",
    "111001",
    "113001",
    "114001",
    "122001",
    "123001",
    "124001",
    "125001",
]


train_records = ["104001", "115001", "118001", "121001", "126001"]
test_records = [
    "100001",
    "100002",
    "103001",
    "103002",
    "103003",
    "105001",
    "111001",
    "113001",
    "114001",
    "122001",
    "123001",
    "124001",
    "125001",
]

x_train = []
x_test = []
y_test = []

for file in recording_names:
    df = pd.read_csv(data_path / (file + "/" + file + "_ANN.csv"), header=None)
    df = df.dropna(subset=[11])
    consensus_labels = df.iloc[:, 11]
    t_start = df.iloc[:, 9]
    t_end = df.iloc[:, 10]

    record_signal = wfdb.rdrecord(data_path / (file + "/" + file + "_ECG"))
    signal_array = record_signal.p_signal.squeeze()
    sig_len = record_signal.sig_len

    full_labels = np.full(sig_len, fill_value=-1, dtype=np.int8)
    for i in range(len(consensus_labels)):
        start_idx = int(t_start.iloc[i]) - 1
        end_idx = int(t_end.iloc[i])

        start_idx = max(0, start_idx)
        end_idx = min(sig_len, end_idx)
        if start_idx >= end_idx:
            continue

        current_label = consensus_labels.iloc[i]
        full_labels[start_idx:end_idx] = current_label

    windows = []
    window_labels = []
    j = 0
    while j < sig_len - 10000 + 1:
        interval_data = signal_array[j : j + 10000]
        interval_labels = full_labels[j : j + 10000]
        bad_label_indices = np.where(np.isin(interval_labels, [-1, 0]))[0]
        if bad_label_indices.size > 0:
            j_new_start_offset = bad_label_indices[-1] + 1
            j += j_new_start_offset
            continue

        values, counts = np.unique(interval_labels, return_counts=True)
        windows.append(np.expand_dims(interval_data, axis=[0, 2]))
        if 3 in values:
            if counts[values.tolist().index(3)] >= 1000:
                window_labels.append(1)
                j += 1000
                continue
        window_labels.append(0)
        j += 1000

    windows = np.concatenate(windows, axis=0)
    window_labels = np.array(window_labels)

    if file in train_records:
        x_train.append(windows)
    else:
        x_test.append(windows)
        y_test.append(window_labels)


x_train = np.concatenate(x_train, axis=0)
x_test = np.concatenate(x_test, axis=0)
y_test = np.concatenate(y_test, axis=0)

np.save("data/brno-university-of-technology-ecg-quality-database/x_train.npy", x_train)
np.save("data/brno-university-of-technology-ecg-quality-database/x_test.npy", x_test)
np.save("data/brno-university-of-technology-ecg-quality-database/y_test.npy", y_test)

# For validation purposes we take only 5% of the test data. This also speeds up the NAS for noise detection.

pos_idx = np.where(y_test == 1)[0]
neg_idx = np.where(y_test == 0)[0]

rnd_pos_idx = np.random.choice(pos_idx, size=int(len(pos_idx) * 0.05), replace=False)
rnd_neg_idx = np.random.choice(neg_idx, size=int(len(neg_idx) * 0.05), replace=False)

final_val_idx = np.concatenate([rnd_pos_idx, rnd_neg_idx])
np.random.shuffle(final_val_idx)

x_test_small = x_test[final_val_idx]
y_test_small = y_test[final_val_idx]

np.save("data/brno-university-of-technology-ecg-quality-database/x_test_small.npy", x_test_small)
np.save("data/brno-university-of-technology-ecg-quality-database/y_test_small.npy", y_test_small)
