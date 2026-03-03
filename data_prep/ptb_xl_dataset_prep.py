# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

# channel order: ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# labels: ["NORM", "MI", "STTC", "CD", "HYP"]


def label_map(label_list):
    final_label = np.array([0, 0, 0, 0, 0])
    if "NORM" in label_list:
        final_label += np.array([1, 0, 0, 0, 0])
    if "HYP" in label_list:
        final_label += np.array([0, 1, 0, 0, 0])
    if "CD" in label_list:
        final_label += np.array([0, 0, 1, 0, 0])
    if "STTC" in label_list:
        final_label += np.array([0, 0, 0, 1, 0])
    if "MI" in label_list:
        final_label += np.array([0, 0, 0, 0, 1])
    return final_label


def label_map_reverse(array):
    final_list = []
    if array[0] == 1:
        final_list.append("NORM")
    if array[1] == 1:
        final_list.append("HYP")
    if array[2] == 1:
        final_list.append("CD")
    if array[3] == 1:
        final_list.append("STTC")
    if array[4] == 1:
        final_list.append("MI")
    return final_list


label_positions = {"HYP": 1, "CD": 2, "STTC": 3, "MI": 4}
conditions = ["HYP", "CD", "STTC", "MI"]
ptb_xl_path = "..." # path to the ptb_xl folder that contains the records500 folder.

# Prep for all ptb_xl train data
# extract filenames and labels from the data csv file
data_list = pd.read_csv(
    "data/ptb_xl/train.csv"
)  # this assumes that you have already used the "ptb_xl_get_csv_files.py" script to generate train, val and test csv files
name_values = data_list["filename_hr"].values
label_values = data_list["diagnostic_superclass"].values

for condition in conditions:
    x_array = []
    y_array = []
    for i in range(len(name_values)):
        data_path = Path(
            ptb_xl_path
            + name_values[i]
        )
        data = wfdb.rdsamp(data_path)[0]
        if not np.isfinite(data).all():
            continue
        label_list = ast.literal_eval(
            label_values[i]
        )  # read string that includes a list as a list
        if len(label_list) == 0:
            continue
        if condition in label_list:
            continue
        x_array.append(np.expand_dims(data, 0))
        y_array.append(
            np.delete(
                np.expand_dims(label_map(label_list), 0),
                label_positions[condition],
                axis=1,
            )
        )

    x_array = np.concatenate(x_array, axis=0)
    y_array = np.concatenate(y_array, axis=0)

    np.save(f"data/ptb_xl/x_train_{condition}.npy", x_array)
    np.save(f"data/ptb_xl/y_train_{condition}.npy", y_array)


# Prep for all ptb_xl val data
# extract filenames and labels from the data csv file
data_list = pd.read_csv(
    "data/ptb_xl/val.csv"
)  # this assumes that you have already used the "ptb_xl_get_csv_files.py" script to generate train, val and test csv files
name_values = data_list["filename_hr"].values
label_values = data_list["diagnostic_superclass"].values

# First the entire val data and all the labels
x_array = []
y_array = []
for i in range(len(name_values)):
    data_path = Path(
        ptb_xl_path
        + name_values[i]
    )
    data = wfdb.rdsamp(data_path)[0]
    if not np.isfinite(data).all():
        continue
    label_list = ast.literal_eval(
        label_values[i]
    )  # read string that includes a list as a list
    if len(label_list) == 0:
        continue
    x_array.append(np.expand_dims(data, 0))
    y_array.append(np.expand_dims(label_map(label_list), 0))

x_array = np.concatenate(x_array, axis=0)
y_array = np.concatenate(y_array, axis=0)
np.save("data/ptb_xl/x_val.npy", x_array)
np.save("data/ptb_xl/y_val_all_labels.npy", y_array)


# Second the val data per condition
for condition in conditions:
    x_array = []
    y_array = []
    for i in range(len(name_values)):
        data_path = Path(
            ptb_xl_path
            + name_values[i]
        )
        data = wfdb.rdsamp(data_path)[0]
        if not np.isfinite(data).all():
            continue
        label_list = ast.literal_eval(
            label_values[i]
        )  # read string that includes a list as a list
        if len(label_list) == 0:
            continue
        if condition in label_list:
            label = np.array([1])
            y_array.append(label)
            continue
        label = np.array([0])
        y_array.append(label)
        x_array.append(np.expand_dims(data, 0))

    x_array = np.concatenate(x_array, axis=0)
    y_array = np.concatenate(y_array, axis=0)

    np.save(f"data/ptb_xl/x_val_{condition}.npy", x_array)
    np.save(f"data/ptb_xl/y_val_{condition}.npy", y_array)


# Prep for all ptb_xl test data
# extract filenames and labels from the data csv file
data_list = pd.read_csv(
    "data/ptb_xl/test.csv"
)  # this assumes that you have already used the "ptb_xl_get_csv_files.py" script to generate train, val and test csv files
name_values = data_list["filename_hr"].values
label_values = data_list["diagnostic_superclass"].values

# First the entire test data and all the labels
x_array = []
y_array = []
for i in range(len(name_values)):
    data_path = Path(
        ptb_xl_path
        + name_values[i]
    )
    data = wfdb.rdsamp(data_path)[0]
    if not np.isfinite(data).all():
        continue
    label_list = ast.literal_eval(
        label_values[i]
    )  # read string that includes a list as a list
    if len(label_list) == 0:
        continue
    x_array.append(np.expand_dims(data, 0))
    y_array.append(np.expand_dims(label_map(label_list), 0))

x_array = np.concatenate(x_array, axis=0)
y_array = np.concatenate(y_array, axis=0)
np.save("data/ptb_xl/x_test.npy", x_array)
np.save("data/ptb_xl/y_test_all_labels.npy", y_array)


# Second the test labels per condition
for condition in conditions:
    y_array = []
    for i in range(len(name_values)):
        data_path = Path(
            ptb_xl_path
            + name_values[i]
        )
        data = wfdb.rdsamp(data_path)[0]
        if not np.isfinite(data).all():
            continue
        label_list = ast.literal_eval(
            label_values[i]
        )  # read string that includes a list as a list
        if len(label_list) == 0:
            continue
        if condition in label_list:
            label = np.array([1])
        else:
            label = np.array([0])
        y_array.append(label)

    y_array = np.concatenate(y_array, axis=0)
    np.save(f"data/ptb_xl/y_test_{condition}.npy", y_array)
