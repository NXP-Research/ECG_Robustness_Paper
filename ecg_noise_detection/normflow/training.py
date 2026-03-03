# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import copy

import config
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training(model, optimizer, train_dataset, norm_clip):
    train_loss = 0
    model.train()
    for batch in train_dataset:
        batch = batch[0].to(device)
        optimizer.zero_grad()
        loss = model.forward_kld(batch)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), norm_clip)
            optimizer.step()
            train_loss += loss.cpu().data.numpy()

    return train_loss / len(train_dataset)


def validation(model, val_dataset):
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in val_dataset:
            batch = batch[0].to(device)
            loss = model.forward_kld(batch)
            val_loss += loss.cpu().data.numpy()

    return val_loss / len(val_dataset)


def evaluate(model, dataset):
    model.eval()
    nll_array = []
    label_array = []
    with torch.no_grad():
        for batch_data, batch_labels in dataset:
            batch_data = batch_data.to(device)
            output = model.log_prob(batch_data, y=None) * -1.0
            nll_array.append(output.cpu().numpy())
            label_array.append(batch_labels.cpu().numpy())
    nll_array = np.concatenate(nll_array)
    label_array = np.concatenate(label_array)
    auc = roc_auc_score(label_array, nll_array)
    return auc


def train(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    val_auc_dataset,
    num_epochs,
    patience,
    val_frequency,
    break_early,
):
    best_auc = -1
    best_weights = copy.deepcopy(model.state_dict())
    lr_status = 0
    patience_checks = patience // val_frequency
    epoch = 0
    norm_clip = config.NORM_CLIP

    while epoch < num_epochs:
        try:
            train_loss_avg = training(model, optimizer, train_dataset, norm_clip)

            if epoch % val_frequency == 0:
                val_loss_avg = validation(model, val_dataset)
                val_auc = evaluate(model, val_auc_dataset)

                print(
                    "epoch:",
                    epoch,
                    "train_loss:",
                    train_loss_avg,
                    "val_loss:",
                    val_loss_avg,
                    "val_auc:",
                    val_auc,
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    best_weights = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience_checks:
                        if lr_status == 0:
                            print("PLATEAU 1!")
                            for g in optimizer.param_groups:
                                g["lr"] = config.LEARNING_RATE_PLATEAU_1
                            patience_counter = 0
                            lr_status = 1
                            model.load_state_dict(best_weights)
                            if break_early:
                                break
                        elif lr_status == 1:
                            print("PLATEAU 2!")
                            for g in optimizer.param_groups:
                                g["lr"] = config.LEARNING_RATE_PLATEAU_2
                            patience_counter = 0
                            lr_status = 2
                            model.load_state_dict(best_weights)
                        elif lr_status == 2:
                            print("PLATEAU 3!")
                            break
            epoch += 1
        except ValueError as e:
            if norm_clip < 1e-10:
                raise e from None
            print(
                f"{e}. Lowering gradient norm clipping threshold for more stability. {norm_clip} --> {0.1 * norm_clip}"
            )
            model.load_state_dict(best_weights)
            norm_clip *= 0.1
            if epoch >= val_frequency:
                epoch -= val_frequency - 1
            continue

    model.load_state_dict(best_weights)
    return best_auc


def instance_normalize(data, epsilon=1e-6):
    mean_instances = np.mean(data, axis=1, keepdims=True)
    std_instances = np.std(data, axis=1, keepdims=True)

    std_instances_safe = np.where(std_instances < epsilon, 1.0, std_instances)

    return (data - mean_instances) / std_instances_safe
