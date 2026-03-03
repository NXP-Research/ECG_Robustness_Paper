# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

# --- PatchTST Configuration ---
PATCHTST_DROPOUT = 0.1
PATCHTST_USE_REVIN = False
PATCHTST_CHANNELS_TOGETHER = True

# --- Data Configuration ---
TIME_STEPS = 512
CHANNELS = 12

# --- Training Hyperparameters ---
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LEARNING_RATE_PLATEAU_1 = 0.0001
LEARNING_RATE_PLATEAU_2 = 0.00001
TRAINING_METHOD = "random_masking"  # any one of: "next_step_prediction", "random_masking", "sequential_masking"
TESTING_METHOD = "sequential_masking"  # any one of: "autoregressive_prediction", "next_step_prediction", "random_masking", "sequential_masking"


# --- Random Masking Hyperparameters ---
MASK_RATIO_TRAIN = 0.05
MASK_RATIO_TEST = 0.05
MASK_REPETITIONS = 100


# --- Sequential Masking Hyperparameters ---
SEQUENCE_LENGTH_TRAIN = 1
SEQUENCE_LENGTH_TEST = 1
SEQUENCE_STRIDE = 1


# --- Autoregressive Prediction Hyperparameters ---
CONTEXT_WINDOW_SIZE = 128
PREDICTION_WINDOW_SIZE = 1
PREDICTION_WINDOW_STRIDE = 1


# --- NAS parameters ---
NAS_EPOCHS = 50
NAS_PATIENCE = 10
NAS_VAL_FREQUENCY = 5
RETRAIN_EPOCHS = 100
RETRAIN_PATIENCE = 20
RETRAIN_VAL_FREQUENCY = 5
