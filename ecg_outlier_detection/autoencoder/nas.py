# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import gc
import multiprocessing as mp
import os
import queue

import config
import numpy as np
import optuna
import tensorflow as tf
from model import Autoencoder, VariationalAutoencoder
from optuna.samplers import RandomSampler
from optuna.study import create_study
from scipy import signal
from training import evaluate, train


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


def create_datasets_for_condition(condition_name):
    x_train = np.load(f"data/ptb_xl/x_train_{condition_name}.npy").astype(np.float32)
    x_train = signal.resample(x=x_train, num=config.TIME_STEPS, axis=1)
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std

    x_val = np.load(f"data/ptb_xl/x_val_{condition_name}.npy").astype(np.float32)
    x_val = signal.resample(x=x_val, num=config.TIME_STEPS, axis=1)
    x_val = (x_val - train_mean) / train_std

    x_val_auc = np.load("data/ptb_xl/x_val.npy").astype(np.float32)
    x_val_auc = signal.resample(x=x_val_auc, num=config.TIME_STEPS, axis=1)
    x_val_auc = (x_val_auc - train_mean) / train_std
    y_val_auc = np.load(f"data/ptb_xl/y_val_{condition_name}.npy").astype(np.float32)

    x_test = np.load("data/ptb_xl/x_test.npy").astype(np.float32)
    x_test = signal.resample(x=x_test, num=config.TIME_STEPS, axis=1)
    x_test = (x_test - train_mean) / train_std
    y_test = np.load(f"data/ptb_xl/y_test_{condition_name}.npy").astype(np.float32)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_val_auc": x_val_auc,
        "y_val_auc": y_val_auc,
        "x_test": x_test,
        "y_test": y_test,
    }


def build_tf_datasets(data_dict):
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(data_dict["x_train"])
        .shuffle(data_dict["x_train"].shape[0], reshuffle_each_iteration=True)
        .batch(config.BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices(data_dict["x_val"])
        .batch(data_dict["x_val"].shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )
    val_auc_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (data_dict["x_val_auc"], data_dict["y_val_auc"])
        )
        .batch(data_dict["x_val_auc"].shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((data_dict["x_test"], data_dict["y_test"]))
        .batch(data_dict["x_test"].shape[0])
        .prefetch(tf.data.AUTOTUNE)
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
        "val_auc": val_auc_dataset,
        "test": test_dataset,
    }


def objective_worker(trial_params, numpy_data, result_queue):
    setup_gpu()
    model_check = None
    try:
        kl_loss_beta = 0.0
        if config.ARCHITECTURE == "VAE":
            kl_loss_beta = trial_params["kl_loss_beta"]

        num_layers = trial_params["num_layers"]
        filter_array = [trial_params[f"filter_{i}"] for i in range(num_layers)]
        kernel_array = [trial_params[f"kernel_{i}"] for i in range(num_layers)]
        stride_array = [trial_params[f"stride_{i}"] for i in range(num_layers)]

        if config.ARCHITECTURE == "VAE":
            model_check = VariationalAutoencoder(
                filter_array=filter_array,
                kernel_array=kernel_array,
                stride_array=stride_array,
                latent_dim=trial_params["latent_dim"],
            )
        elif config.ARCHITECTURE == "AE":
            model_check = Autoencoder(
                filter_array=filter_array,
                kernel_array=kernel_array,
                stride_array=stride_array,
                latent_dim=trial_params["latent_dim"],
            )
        else:
            raise NotImplementedError(
                f"Architecture '{config.ARCHITECTURE}' not supported for NAS."
            )

        build_input = tf.keras.Input(
            shape=(config.TIME_STEPS, config.CHANNELS), dtype=tf.float32
        )
        _ = model_check(build_input)
        param_count = model_check.count_params()

        MAX_PARAMS = 512000
        if param_count > MAX_PARAMS:
            raise optuna.exceptions.TrialPruned(
                f"Exceeded parameter limit: {param_count} > {MAX_PARAMS}"
            )

        avg_auc = 0
        tf_datasets = build_tf_datasets(numpy_data)

        for i in range(3):
            model, optimizer = None, None
            try:
                kl_loss_beta = 0.0
                if config.ARCHITECTURE == "VAE":
                    kl_loss_beta = trial_params["kl_loss_beta"]

                num_layers = trial_params["num_layers"]
                filter_array = [trial_params[f"filter_{i}"] for i in range(num_layers)]
                kernel_array = [trial_params[f"kernel_{i}"] for i in range(num_layers)]
                stride_array = [trial_params[f"stride_{i}"] for i in range(num_layers)]

                if config.ARCHITECTURE == "VAE":
                    model = VariationalAutoencoder(
                        filter_array=filter_array,
                        kernel_array=kernel_array,
                        stride_array=stride_array,
                        latent_dim=trial_params["latent_dim"],
                    )
                elif config.ARCHITECTURE == "AE":
                    model = Autoencoder(
                        filter_array=filter_array,
                        kernel_array=kernel_array,
                        stride_array=stride_array,
                        latent_dim=trial_params["latent_dim"],
                    )
                else:
                    raise NotImplementedError(
                        f"Architecture '{config.ARCHITECTURE}' not supported for NAS."
                    )
                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=config.LEARNING_RATE
                )
                auc = train(
                    model,
                    optimizer,
                    tf_datasets["train"],
                    tf_datasets["val"],
                    tf_datasets["val_auc"],
                    num_epochs=config.NAS_EPOCHS,
                    patience=config.NAS_PATIENCE,
                    val_frequency=config.NAS_VAL_FREQUENCY,
                    kl_loss_beta=kl_loss_beta,
                    break_early=True,
                )
                avg_auc += auc
            finally:
                del model
                del optimizer
                clear_session()

        result_queue.put((avg_auc / 3, param_count))

    except optuna.exceptions.TrialPruned as e:
        print(f"Trial pruned: {e}")
        result_queue.put("PRUNED")
    except tf.errors.ResourceExhaustedError:
        print("GPU memory exhausted, pruning trial")
        result_queue.put("PRUNED")
    except Exception as e:
        print(f"An unexpected error occurred in trial: {e}")
        result_queue.put("FAIL")
    finally:
        del model_check
        clear_session()


def retrain_worker(params, datasets_numpy, result_queue):
    setup_gpu()
    model, optimizer = None, None
    try:
        kl_loss_beta = 0.0
        if config.ARCHITECTURE == "VAE":
            kl_loss_beta = params["kl_loss_beta"]

        num_layers = params["num_layers"]
        filter_array = [params[f"filter_{i}"] for i in range(num_layers)]
        kernel_array = [params[f"kernel_{i}"] for i in range(num_layers)]
        stride_array = [params[f"stride_{i}"] for i in range(num_layers)]

        if config.ARCHITECTURE == "VAE":
            model = VariationalAutoencoder(
                filter_array=filter_array,
                kernel_array=kernel_array,
                stride_array=stride_array,
                latent_dim=params["latent_dim"],
            )
        elif config.ARCHITECTURE == "AE":
            model = Autoencoder(
                filter_array=filter_array,
                kernel_array=kernel_array,
                stride_array=stride_array,
                latent_dim=params["latent_dim"],
            )
        else:
            raise ValueError(
                f"Architecture '{config.ARCHITECTURE}' not supported for retraining."
            )

        build_input = tf.keras.Input(
            shape=(config.TIME_STEPS, config.CHANNELS), dtype=tf.float32
        )
        _ = model(build_input)
        param_count = model.count_params()

        tf_datasets = build_tf_datasets(datasets_numpy)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE)

        train(
            model,
            optimizer,
            tf_datasets["train"],
            tf_datasets["val"],
            tf_datasets["val_auc"],
            num_epochs=config.RETRAIN_EPOCHS,
            patience=config.RETRAIN_PATIENCE,
            val_frequency=config.RETRAIN_VAL_FREQUENCY,
            kl_loss_beta=kl_loss_beta,
            break_early=False,
        )

        test_auc = evaluate(model, tf_datasets["test"], training=False)
        result_queue.put((test_auc, param_count))

    except Exception as e:
        print(f"An unexpected error occurred during retraining: {e}")
        result_queue.put("FAIL")
    finally:
        del model
        del optimizer
        clear_session()


def retrain_pareto(pareto_front, numpy_data):
    for i, trial in enumerate(pareto_front):
        print(
            f"\nRetraining Pareto front model {i + 1}/{len(pareto_front)} with params: {trial.params}"
        )
        run_aucs = []
        final_param_count = None

        for j in range(5):
            result_queue = mp.Queue()
            p = mp.Process(
                target=retrain_worker,
                args=(trial.params, numpy_data, result_queue),
            )
            p.start()
            p.join()

            try:
                result = result_queue.get_nowait()
                if result == "FAIL":
                    print(f"    Run {j + 1}/5 failed.")
                else:
                    test_auc, param_count = result
                    final_param_count = param_count
                    run_aucs.append(test_auc)
                    print(f"    Run {j + 1}/5 -> Test AUC: {test_auc:.4f}")
            except queue.Empty:
                print(f"    Run {j + 1}/5 failed (process crash).")

        if len(run_aucs) == 5:
            mean_auc = np.mean(run_aucs)
            std_auc = np.std(run_aucs)

            with open(
                f"ecg_outlier_detection/autoencoder/results_{config.ARCHITECTURE.lower()}_{CONDITION}.txt",
                "a",
            ) as log_file:
                log_file.write(f"\nModel {i + 1}:\n")
                log_file.write(f"  Architecture: {config.ARCHITECTURE}\n")
                log_file.write(f"  Parameters: {trial.params}\n")
                log_file.write(f"  Parameter Count: {final_param_count}\n")
                log_file.write(f"  Test AUC (5 runs): {mean_auc} +/- {std_auc}\n")
                log_file.flush()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    CONDITION = "HYP"
    N_SUCCESSFUL_TRIALS = 3

    print("Loading and preparing the PTB-XL dataset as NumPy arrays.")
    numpy_data = create_datasets_for_condition(CONDITION)
    print("Dataset loaded into memory.")

    db_name_part = f"autoencoder_{config.ARCHITECTURE.lower()}_{CONDITION}"
    study = create_study(
        study_name=f"nas_{db_name_part}",
        storage=f"sqlite:///ecg_outlier_detection/autoencoder/nas_{db_name_part}.db",
        directions=["maximize", "minimize"],
        sampler=RandomSampler(),
        load_if_exists=True,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print(f"Starting NAS optimization to find {N_SUCCESSFUL_TRIALS} successful trials.")

    while (
        len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        < N_SUCCESSFUL_TRIALS
    ):
        total_trials_so_far = len(study.get_trials(deepcopy=False))
        print(f"\n--- Starting Trial {total_trials_so_far + 1} ---")

        trial = study.ask()

        params = {}
        num_layers = trial.suggest_int("num_layers", 2, 8)
        params["num_layers"] = num_layers
        params["latent_dim"] = trial.suggest_categorical(
            "latent_dim", [12, 16, 32, 64, 128, 256, 512, 1024]
        )

        for i in range(num_layers):
            params[f"filter_{i}"] = trial.suggest_categorical(
                f"filter_{i}", [12, 16, 24, 32, 64, 128, 256, 512, 1024]
            )
            params[f"kernel_{i}"] = trial.suggest_categorical(
                f"kernel_{i}", [3, 5, 7, 9, 11]
            )
            params[f"stride_{i}"] = trial.suggest_categorical(f"stride_{i}", [1, 2])

        if config.ARCHITECTURE == "VAE":
            params["kl_loss_beta"] = trial.suggest_categorical(
                "kl_loss_beta", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0]
            )

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        result_queue = mp.Queue()
        p = mp.Process(target=objective_worker, args=(params, numpy_data, result_queue))
        p.start()
        p.join()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        try:
            result = result_queue.get()
            if result == "PRUNED":
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                print(f"Trial {trial.number} pruned.")
            elif result == "FAIL":
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                print(f"Trial {trial.number} failed.")
            else:
                auc_val, params_val = result
                study.tell(trial, values=[auc_val, params_val])
                print(
                    f"Trial {trial.number} finished successfully. AUC: {auc_val:.4f}, Params: {params_val}"
                )
        except queue.Empty:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            print(f"Trial {trial.number} failed (process crash).")

        completed_count = len(
            study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        )
        print(f"Progress: {completed_count} / {N_SUCCESSFUL_TRIALS} successful trials.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("\nNAS optimization finished.")
    pareto_front = study.best_trials
    print("\nPareto Front:")
    for i, t in enumerate(pareto_front):
        if t.state == optuna.trial.TrialState.COMPLETE:
            print(
                f"  Model {i + 1}: Params={t.params}, AUC={t.values[0]:.4f}, ParamCount={t.values[1]}"
            )

    print("\nRetraining Pareto front models.")
    retrain_pareto(pareto_front, numpy_data)
