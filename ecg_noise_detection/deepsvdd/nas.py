# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import copy
import gc
import multiprocessing as mp
import os
import queue

import config
import numpy as np
import optuna
import tensorflow as tf
from model import ResNet
from optuna.samplers import RandomSampler
from optuna.study import create_study
from scipy import signal
from training import evaluate, get_center, instance_normalize, train


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


def create_datasets():
    base_path = "data/brno-university-of-technology-ecg-quality-database"

    x_train = np.load(f"{base_path}/x_train.npy").astype(np.float32)
    x_train = signal.resample(x=x_train, num=config.TIME_STEPS, axis=1)
    x_train = instance_normalize(x_train)

    x_val_auc = np.load(f"{base_path}/x_test_small.npy").astype(np.float32)
    x_val_auc = signal.resample(x=x_val_auc, num=config.TIME_STEPS, axis=1)
    x_val_auc = instance_normalize(x_val_auc)
    y_val_auc = np.load(f"{base_path}/y_test_small.npy").astype(np.float32)

    x_val = copy.deepcopy(x_val_auc[y_val_auc == 0])

    x_test = np.load(f"{base_path}/x_test.npy").astype(np.float32)
    x_test = signal.resample(x=x_test, num=config.TIME_STEPS, axis=1)
    x_test = instance_normalize(x_test)
    y_test = np.load(f"{base_path}/y_test.npy").astype(np.float32)

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
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_auc_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (data_dict["x_val_auc"], data_dict["y_val_auc"])
        )
        .batch(1024)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((data_dict["x_test"], data_dict["y_test"]))
        .batch(1024)
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
        model_check = ResNet(**trial_params)

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
        train_data_count = numpy_data["x_train"].shape[0]

        for i in range(3):
            model, optimizer = None, None
            try:
                model = ResNet(**trial_params)

                optimizer = tf.keras.optimizers.AdamW(
                    learning_rate=config.LEARNING_RATE
                )
                center = tf.convert_to_tensor(
                    get_center(
                        model,
                        tf_datasets["train"],
                        train_data_count,
                        trial_params["latent_dim"],
                    ),
                    dtype=tf.float32,
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
                    break_early=True,
                    center=center,
                )
                avg_auc += auc
            finally:
                del model
                del optimizer
                del center
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

        build_input = tf.keras.Input(
            shape=(config.TIME_STEPS, config.CHANNELS), dtype=tf.float32
        )
        _ = model(build_input)
        param_count = model.count_params()

        tf_datasets = build_tf_datasets(datasets_numpy)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=config.LEARNING_RATE)

        train_data_count = datasets_numpy["x_train"].shape[0]
        center = tf.convert_to_tensor(
            get_center(
                model, tf_datasets["train"], train_data_count, params["latent_dim"]
            ),
            dtype=tf.float32,
        )

        train(
            model,
            optimizer,
            tf_datasets["train"],
            tf_datasets["val"],
            tf_datasets["val_auc"],
            num_epochs=config.RETRAIN_EPOCHS,
            patience=config.RETRAIN_PATIENCE,
            val_frequency=config.RETRAIN_VAL_FREQUENCY,
            break_early=False,
            center=center,
        )

        test_auc = evaluate(model, tf_datasets["test"], center, training=False)
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
                "ecg_noise_detection/deepsvdd/results.txt",
                "a",
            ) as log_file:
                log_file.write(f"\nModel {i + 1}:\n")
                log_file.write(f"  Parameters: {trial.params}\n")
                log_file.write(f"  Parameter Count: {final_param_count}\n")
                log_file.write(f"  Test AUC (5 runs): {mean_auc} +/- {std_auc}\n")
                log_file.flush()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    N_SUCCESSFUL_TRIALS = 5

    print("Loading and preparing the BRNO dataset as NumPy arrays.")
    numpy_data = create_datasets()
    print("Dataset loaded into memory.")

    db_name_part = "deepsvdd"
    study = create_study(
        study_name=f"nas_{db_name_part}",
        storage=f"sqlite:///ecg_noise_detection/deepsvdd/nas_{db_name_part}.db",
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
        num_layers = trial.suggest_int("num_layers", 2, 8)
        initial_num_filter = trial.suggest_categorical(
            "initial_num_filter", [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
        )
        initial_kernel_size = trial.suggest_categorical(
            "initial_kernel_size", [3, 5, 7, 9, 11]
        )
        initial_stride = trial.suggest_categorical("initial_stride", [1, 2])

        num_filters_list = []
        kernel_size_list = []
        strides_list = []

        for i in range(num_layers):
            num_filters_list.append(
                trial.suggest_categorical(
                    f"filters_{i}", [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
                )
            )

            kernel_size_list.append(
                trial.suggest_categorical(f"kernel_{i}", [3, 5, 7, 9, 11])
            )

            strides_list.append(trial.suggest_categorical(f"stride_{i}", [1, 2]))

        params = {
            "initial_num_filter": initial_num_filter,
            "initial_kernel_size": initial_kernel_size,
            "initial_stride": initial_stride,
            "num_filters_list": num_filters_list,
            "kernel_size_list": kernel_size_list,
            "strides_list": strides_list,
            "latent_dim": trial.suggest_categorical(
                "latent_dim", [8, 16, 32, 64, 128, 256, 512, 1024]
            ),
        }

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
