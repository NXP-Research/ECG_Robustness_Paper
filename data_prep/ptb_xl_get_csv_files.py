# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: CC-BY-4.0

import ast
from pathlib import Path

import pandas as pd

# The following code was adapted from PTB-XL
# Source: https://physionet.org/content/ptb-xl/1.0.3/example_physionet.py
# Licensed under the Creative Commons Attribution 4.0 International Public License: https://creativecommons.org/licenses/by/4.0/legalcode.en

path = Path(
    ...
)  # path to the ptb-xl folder that you downloaded. It should contain "ptbxl_database.csv" and "scp_statements.csv"
sampling_rate = 500

# load and convert annotation data
Y = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path / "scp_statements.csv", index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


# Apply diagnostic superclass
Y["diagnostic_superclass"] = Y.scp_codes.apply(aggregate_diagnostic)

Y = Y.set_index("filename_hr")

Y_val = Y[Y.strat_fold == 9]
Y_test = Y[Y.strat_fold == 10]
Y_train = Y[Y.strat_fold != 9]
Y_train = Y_train[Y_train.strat_fold != 10]

Y_val = Y_val[["diagnostic_superclass"]]
Y_test = Y_test[["diagnostic_superclass"]]
Y_train = Y_train[["diagnostic_superclass"]]

Y_train.to_csv("data/ptb_xl/train.csv", index=True)
Y_val.to_csv("data/ptb_xl/val.csv", index=True)
Y_test.to_csv("data/ptb_xl/test.csv", index=True)
