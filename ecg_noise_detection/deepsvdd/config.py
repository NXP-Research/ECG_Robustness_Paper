# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

# --- Data Configuration ---
TIME_STEPS = 512
CHANNELS = 1

# --- Deep SVDD Configuration ---
LATENT_DIM = 128  # Dimensionality of the latent space for the hypersphere center

# --- Training Hyperparameters ---
BATCH_SIZE = 256
LEARNING_RATE = 0.001
LEARNING_RATE_PLATEAU_1 = 0.0001
LEARNING_RATE_PLATEAU_2 = 0.00001

# --- NAS parameters ---
NAS_EPOCHS = 50
NAS_PATIENCE = 10
NAS_VAL_FREQUENCY = 5
RETRAIN_EPOCHS = 100
RETRAIN_PATIENCE = 20
RETRAIN_VAL_FREQUENCY = 5
