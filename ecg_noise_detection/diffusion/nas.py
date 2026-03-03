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
from model import (
    GaussianDiffusion1D,
    Trainer1D,
    Unet1D,
    get_model_size,
    instance_normalize,
)
from optuna.samplers import RandomSampler
from optuna.study import create_study
from scipy import signal
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_datasets():
    base_path = "data/brno-university-of-technology-ecg-quality-database"

    x_train = np.load(f"{base_path}/x_train.npy").astype(np.float32)
    x_train = signal.resample(x=x_train, num=config.TIME_STEPS, axis=1)
    x_train = instance_normalize(x_train)
    train_min = np.min(x_train)
    train_max = np.max(x_train)
    x_train = (x_train - train_min) / (train_max - train_min)

    x_val_auc = np.load(f"{base_path}/x_test_small.npy").astype(np.float32)
    x_val_auc = signal.resample(x=x_val_auc, num=config.TIME_STEPS, axis=1)
    x_val_auc = instance_normalize(x_val_auc)
    x_val_auc = (x_val_auc - train_min) / (train_max - train_min)
    y_val_auc = np.load(f"{base_path}/y_test_small.npy").astype(np.float32)

    x_val = copy.deepcopy(x_val_auc[y_val_auc == 0])

    x_test = np.load(f"{base_path}/x_test.npy").astype(np.float32)
    x_test = signal.resample(x=x_test, num=config.TIME_STEPS, axis=1)
    x_test = instance_normalize(x_test)
    x_test = (x_test - train_min) / (train_max - train_min)
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
        drop_last=True,
    )
    x_val_dataloader = DataLoader(
        TensorDataset(x_val), batch_size=1024, shuffle=False, drop_last=False
    )
    xy_val_auc_dataloader = DataLoader(
        TensorDataset(x_val_auc, y_val_auc),
        batch_size=1024,
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
        unet_check = Unet1D(
            dim=trial_params["dim"],
            dim_mults=trial_params["dim_mults"],
            channels=config.CHANNELS,
            attn_dim_head=trial_params["attn_dim_head"],
            attn_heads=trial_params["attn_heads"],
        )

        model_check = GaussianDiffusion1D(unet_check, seq_length=config.TIME_STEPS)

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
                unet = Unet1D(
                    dim=trial_params["dim"],
                    dim_mults=trial_params["dim_mults"],
                    channels=config.CHANNELS,
                    attn_dim_head=trial_params["attn_dim_head"],
                    attn_heads=trial_params["attn_heads"],
                )

                model = GaussianDiffusion1D(
                    unet,
                    seq_length=config.TIME_STEPS,
                    timesteps=trial_params["train_noise_steps"],
                    objective=trial_params["objective"],
                    auto_normalize=True,
                )

                trainer = Trainer1D(
                    diffusion_model=model,
                    dataset=torch_datasets["train"],
                    eval_dataset=torch_datasets["val_auc"],
                    test_dataset=torch_datasets["test"],
                    train_batch_size=config.BATCH_SIZE,
                    train_lr=config.LEARNING_RATE,
                    train_num_steps=numpy_data["x_train"].shape[0]
                    // config.BATCH_SIZE
                    // config.GRADIENT_ACCUMULATE_EVERY
                    * config.NAS_EPOCHS,  # total training steps
                    gradient_accumulate_every=config.GRADIENT_ACCUMULATE_EVERY,  # gradient accumulation steps
                    ema_decay=config.EMA_DECAY,  # exponential moving average decay
                    amp=False,  # turn on mixed precision
                    evaluate_every=numpy_data["x_train"].shape[0]
                    // config.BATCH_SIZE
                    // config.GRADIENT_ACCUMULATE_EVERY
                    * config.NAS_VAL_FREQUENCY,
                    anomaly_noise_steps=trial_params["anomaly_noise_steps"],
                    max_grad_norm=config.NORM_CLIP,
                    break_early=True,
                    patience=config.NAS_PATIENCE,
                )
                auc, _ = trainer.train()
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
        dim_mults = []

        for j in range(params["num_layers"]):
            dim_mults.append(params[f"dim_mult_{j}"])

        anomaly_noise_steps = int(
            params["train_noise_steps"] * params["anomaly_noise_step_multiplier"]
        )

        unet = Unet1D(
            dim=params["dim"],
            dim_mults=dim_mults,
            channels=config.CHANNELS,
            attn_dim_head=params["attn_dim_head"],
            attn_heads=params["attn_heads"],
        )

        model = GaussianDiffusion1D(
            unet,
            seq_length=config.TIME_STEPS,
            timesteps=params["train_noise_steps"],
            objective=params["objective"],
            auto_normalize=True,
        )

        param_count = get_model_size(model)
        torch_datasets = build_torch_datasets(datasets_numpy)

        trainer = Trainer1D(
            diffusion_model=model,
            dataset=torch_datasets["train"],
            eval_dataset=torch_datasets["val_auc"],
            test_dataset=torch_datasets["test"],
            train_batch_size=config.BATCH_SIZE,
            train_lr=config.LEARNING_RATE,
            train_num_steps=datasets_numpy["x_train"].shape[0]
            // config.BATCH_SIZE
            // config.GRADIENT_ACCUMULATE_EVERY
            * config.RETRAIN_EPOCHS,  # total training steps
            gradient_accumulate_every=config.GRADIENT_ACCUMULATE_EVERY,  # gradient accumulation steps
            ema_decay=config.EMA_DECAY,  # exponential moving average decay
            amp=False,  # turn on mixed precision
            evaluate_every=datasets_numpy["x_train"].shape[0]
            // config.BATCH_SIZE
            // config.GRADIENT_ACCUMULATE_EVERY
            * config.RETRAIN_VAL_FREQUENCY,
            anomaly_noise_steps=anomaly_noise_steps,
            max_grad_norm=config.NORM_CLIP,
            break_early=False,
            patience=config.RETRAIN_PATIENCE,
        )
        _, test_auc = trainer.train()
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
                "ecg_noise_detection/diffusion/results.txt",
                "a",
            ) as log_file:
                log_file.write(f"\nModel {i + 1}:\n")
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

    db_name_part = "diffusion"
    study = create_study(
        study_name=f"nas_{db_name_part}",
        storage=f"sqlite:///ecg_noise_detection/diffusion/nas_{db_name_part}.db",
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

        train_noise_steps = trial.suggest_int("train_noise_steps", 100, 1000)
        anomaly_noise_step_multiplier = trial.suggest_categorical(
            "anomaly_noise_step_multiplier",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        anomaly_noise_steps = int(train_noise_steps * anomaly_noise_step_multiplier)
        num_layers = trial.suggest_int("num_layers", 1, 10)
        dim = trial.suggest_categorical("dim", [12, 16, 24, 32, 64, 128, 256])
        attn_dim_head = trial.suggest_categorical(
            "attn_dim_head", [12, 16, 24, 32, 64, 128, 256]
        )
        attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 6, 8])
        objective = trial.suggest_categorical(
            "objective", ["pred_noise", "pred_x0", "pred_v"]
        )

        dim_mults = []

        for j in range(num_layers):
            dim_mults.append(trial.suggest_categorical(f"dim_mult_{j}", [1, 2, 4, 8]))

        params = {
            "train_noise_steps": train_noise_steps,
            "anomaly_noise_steps": anomaly_noise_steps,
            "dim_mults": dim_mults,
            "dim": dim,
            "num_layers": num_layers,
            "attn_dim_head": attn_dim_head,
            "attn_heads": attn_heads,
            "objective": objective,
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
