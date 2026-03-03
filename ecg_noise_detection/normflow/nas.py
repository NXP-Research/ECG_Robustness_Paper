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
import torch
from model import get_model_size, get_multiscale_model
from optuna.samplers import RandomSampler
from optuna.study import create_study
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset
from training import evaluate, instance_normalize, train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def build_torch_datasets(data_dict):
    x_train = torch.tensor(data_dict["x_train"], dtype=torch.float32).permute(0, 2, 1)
    x_val = torch.tensor(data_dict["x_val"], dtype=torch.float32).permute(0, 2, 1)
    x_val_auc = torch.tensor(data_dict["x_val_auc"], dtype=torch.float32).permute(
        0, 2, 1
    )
    y_val_auc = torch.tensor(data_dict["y_val_auc"], dtype=torch.float32)
    x_test = torch.tensor(data_dict["x_test"], dtype=torch.float32).permute(0, 2, 1)
    y_test = torch.tensor(data_dict["y_test"], dtype=torch.float32)

    x_train_dataloader = DataLoader(
        TensorDataset(x_train),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    x_val_dataloader = DataLoader(
        TensorDataset(x_val), batch_size=x_val.shape[0], shuffle=False, drop_last=False
    )
    xy_val_auc_dataloader = DataLoader(
        TensorDataset(x_val_auc, y_val_auc),
        batch_size=x_val_auc.shape[0],
        shuffle=False,
        drop_last=False,
    )
    xy_test_dataloader = DataLoader(
        TensorDataset(x_test, y_test), batch_size=1024, shuffle=False, drop_last=False
    )

    return {
        "train": x_train_dataloader,
        "val": x_val_dataloader,
        "val_auc": xy_val_auc_dataloader,
        "test": xy_test_dataloader,
    }


def objective_worker(trial_params, numpy_data, result_queue):
    model_check = None
    try:
        model_check = get_multiscale_model(**trial_params)

        param_count = get_model_size(model_check)

        MAX_PARAMS = 512000
        if param_count > MAX_PARAMS:
            raise optuna.exceptions.TrialPruned(
                f"Exceeded parameter limit: {param_count} > {MAX_PARAMS}"
            )

        avg_auc = 0
        torch_datasets = build_torch_datasets(numpy_data)

        for i in range(3):
            model, optimizer = None, None
            try:
                model = get_multiscale_model(**trial_params).to(device)

                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config.LEARNING_RATE
                )
                auc = train(
                    model,
                    optimizer,
                    torch_datasets["train"],
                    torch_datasets["val"],
                    torch_datasets["val_auc"],
                    num_epochs=config.NAS_EPOCHS,
                    patience=config.NAS_PATIENCE,
                    val_frequency=config.NAS_VAL_FREQUENCY,
                    break_early=True,
                )
                avg_auc += auc
            finally:
                del model
                del optimizer
                torch.cuda.empty_cache()
                gc.collect()

        result_queue.put((avg_auc / 3, param_count))

    except optuna.exceptions.TrialPruned as e:
        print(f"Trial pruned: {e}")
        result_queue.put("PRUNED")
    except torch.cuda.OutOfMemoryError:
        print("GPU memory exhausted, pruning trial")
        result_queue.put("PRUNED")
    except ValueError as e:
        print(
            f"Network too unstable even with aggressive gradient clipping: {e}. Pruning trial."
        )
        result_queue.put("PRUNED")
    except Exception as e:
        print(f"An unexpected error occurred in trial: {e}")
        result_queue.put("FAIL")
    finally:
        del model_check
        torch.cuda.empty_cache()
        gc.collect()


def retrain_worker(params, datasets_numpy, result_queue):
    model, optimizer = None, None
    try:
        if config.ARCHITECTURE == "MLP":
            network = params["network"]
            num_levels = params["num_levels"]
            num_layers = params["num_layers"]
            num_blocks = []
            filter_list = []
            kernel_list = params["kernel_list"]
            split_ratio_list = []
            squeeze_ratio_list = []
            split_mode = params["split_mode"]
            permute_mode = params["permute_mode"]
            latent_split_mode = params["latent_split_mode"]
            param_dim = params["param_dim"]

            for j in range(num_layers):
                filter_list.append(params[f"filter_{j}"])
            for i in range(num_levels):
                num_blocks.append(params[f"num_blocks_{i}"])
                split_ratio_list.append(params[f"split_ratio_list_{i}"])
                squeeze_ratio_list.append(params[f"squeeze_ratio_list_{i}"])

            model = get_multiscale_model(
                filter_list,
                kernel_list,
                num_blocks,
                params["input_shape"],
                num_levels,
                split_ratio_list,
                squeeze_ratio_list,
                split_mode,
                permute_mode,
                latent_split_mode,
                network,
                param_dim,
            ).to(device)
        elif config.ARCHITECTURE == "CNN":
            network = "CNN"
            num_levels = params["num_levels"]
            num_layers = params["num_layers"]
            num_blocks = []
            filter_list = []
            kernel_list = []
            split_ratio_list = []
            squeeze_ratio_list = []
            split_mode = params["split_mode"]
            permute_mode = params["permute_mode"]
            latent_split_mode = params["latent_split_mode"]
            param_dim = "channel"

            for j in range(num_layers):
                filter_list.append(params[f"filter_{j}"])
                kernel_list.append(params[f"kernel_{j}"])
            kernel_list.append(params[f"kernel_{num_layers}"])
            for i in range(num_levels):
                num_blocks.append(params[f"num_blocks_{i}"])
                split_ratio_list.append(params[f"split_ratio_{i}"])
                squeeze_ratio_list.append(params[f"squeeze_ratio_{i}"])

            model = get_multiscale_model(
                filter_list,
                kernel_list,
                num_blocks,
                (config.CHANNELS, config.TIME_STEPS),
                num_levels,
                split_ratio_list,
                squeeze_ratio_list,
                split_mode,
                permute_mode,
                latent_split_mode,
                network,
                param_dim,
            ).to(device)
        else:
            raise NotImplementedError(
                f"Architecture '{config.ARCHITECTURE}' not supported for NAS."
            )

        param_count = get_model_size(model)

        torch_datasets = build_torch_datasets(datasets_numpy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

        train(
            model,
            optimizer,
            torch_datasets["train"],
            torch_datasets["val"],
            torch_datasets["val_auc"],
            num_epochs=config.RETRAIN_EPOCHS,
            patience=config.RETRAIN_PATIENCE,
            val_frequency=config.RETRAIN_VAL_FREQUENCY,
            break_early=False,
        )
        test_auc = evaluate(model, torch_datasets["test"])
        result_queue.put((test_auc, param_count))

    except Exception as e:
        print(f"An unexpected error occurred during retraining: {e}")
        result_queue.put("FAIL")
    finally:
        del model
        del optimizer
        torch.cuda.empty_cache()
        gc.collect()


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
                f"ecg_noise_detection/normflow/results_{config.ARCHITECTURE.lower()}.txt",
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

    N_SUCCESSFUL_TRIALS = 3

    print("Loading and preparing the BRNO dataset as NumPy arrays.")
    numpy_data = create_datasets()
    print("Dataset loaded into memory.")

    db_name_part = f"normflow_{config.ARCHITECTURE.lower()}"
    study = create_study(
        study_name=f"nas_{db_name_part}",
        storage=f"sqlite:///ecg_noise_detection/normflow/nas_{db_name_part}.db",
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

        num_levels = trial.suggest_int("num_levels", 1, 5)
        num_layers = trial.suggest_int("num_layers", 1, 5)

        if config.ARCHITECTURE == "MLP":
            network = "MLP"
            num_blocks = []
            filter_list = []
            kernel_list = None
            split_ratio_list = []
            squeeze_ratio_list = []

            split_mode = "checkerboard"
            permute_mode = "time"
            latent_split_mode = "time"
            param_dim = "time"

            for j in range(num_layers):
                filter_list.append(
                    trial.suggest_categorical(
                        f"filter_{j}", [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
                    )
                )

            for i in range(num_levels):
                num_blocks.append(trial.suggest_int(f"num_blocks_{i}", 1, 10))

                split_ratio_list.append(
                    trial.suggest_categorical(f"split_ratio_{i}", [0.25, 0.5, 0.75])
                )

                squeeze_ratio_list.append(1)

            params = {
                "filter_list": filter_list,
                "kernel_list": kernel_list,
                "num_blocks": num_blocks,
                "input_shape": (config.CHANNELS, config.TIME_STEPS),
                "num_levels": num_levels,
                "split_ratio_list": split_ratio_list,
                "squeeze_ratio_list": squeeze_ratio_list,
                "network": network,
                "split_mode": split_mode,
                "latent_split_mode": latent_split_mode,
                "permute_mode": permute_mode,
                "param_dim": param_dim,
            }
        elif config.ARCHITECTURE == "CNN":
            network = "CNN"
            num_blocks = []
            filter_list = []
            kernel_list = []
            split_ratio_list = []
            squeeze_ratio_list = []

            split_mode = trial.suggest_categorical(
                "split_mode", ["channel", "checkerboard"]
            )  # channel, checkerboard
            permute_mode = trial.suggest_categorical(
                "permute_mode", ["channel", "time"]
            )  # channel, time
            latent_split_mode = trial.suggest_categorical(
                "latent_split_mode", ["channel", "time"]
            )  # channel, time
            param_dim = "channel"

            for j in range(num_layers):
                filter_list.append(
                    trial.suggest_categorical(
                        f"filter_{j}", [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
                    )
                )

                kernel_list.append(
                    trial.suggest_categorical(f"kernel_{j}", [3, 5, 7, 9, 11])
                )

            kernel_list.append(
                trial.suggest_categorical(f"kernel_{num_layers}", [3, 5, 7, 9, 11])
            )

            for i in range(num_levels):
                num_blocks.append(trial.suggest_int(f"num_blocks_{i}", 1, 10))

                split_ratio_list.append(
                    trial.suggest_categorical(f"split_ratio_{i}", [0.25, 0.5, 0.75])
                )

                squeeze_ratio_list.append(
                    trial.suggest_categorical(f"squeeze_ratio_{i}", [1, 2, 4])
                )

            params = {
                "filter_list": filter_list,
                "kernel_list": kernel_list,
                "num_blocks": num_blocks,
                "input_shape": (config.CHANNELS, config.TIME_STEPS),
                "num_levels": num_levels,
                "split_ratio_list": split_ratio_list,
                "squeeze_ratio_list": squeeze_ratio_list,
                "network": network,
                "split_mode": split_mode,
                "latent_split_mode": latent_split_mode,
                "permute_mode": permute_mode,
                "param_dim": param_dim,
            }

        else:
            raise NotImplementedError(
                f"Architecture '{config.ARCHITECTURE}' not supported for NAS."
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
