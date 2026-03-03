# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import random

import numpy as np
import pywt
import wfdb
from tqdm import tqdm
from wfdb import processing


def wavelet_denoise(data, wavelet="db6", decomp_level=11):
    coeffs = pywt.wavedec(data, wavelet, level=decomp_level)
    sigma = np.median(np.abs(coeffs[1] - np.median(coeffs[1]))) / 0.6745
    N = len(data)
    threshold = sigma * np.sqrt(2 * np.log(N))

    new_coeffs = coeffs.copy()
    for i in range(1, len(coeffs)):
        new_coeffs[i] = pywt.threshold(coeffs[i], value=threshold, mode="soft")

    denoised_signal = pywt.waverec(new_coeffs, wavelet)
    noise_signal = data - denoised_signal

    return denoised_signal, noise_signal


def calculate_N(noise_signal, fs):
    samples_per_chunk = int(fs)
    num_chunks = len(noise_signal) // samples_per_chunk

    if num_chunks < 1:
        # print(
        #     "Warning: Signal is under 1s. Returning the squared RMS of entire noise signal"
        # )
        return np.square(
            np.sqrt(np.mean(np.square(noise_signal - np.mean(noise_signal))))
        )

    rms_values = []
    for i in range(num_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = noise_signal[start:end]
        rms = np.sqrt(np.mean(np.square(chunk - np.mean(chunk))))
        rms_values.append(rms)

    if len(rms_values) < 3:
        # print(
        #     "Warning: Not enough rms values for percentile filtering. Returning the squared mean without filtering."
        # )
        return np.square(np.mean(rms_values))

    lower_bound = np.percentile(rms_values, 5)
    upper_bound = np.percentile(rms_values, 95)
    filtered_rms = [rms for rms in rms_values if lower_bound <= rms <= upper_bound]

    if not filtered_rms:
        # print(
        #     "Warning: Filtering removed all rms values. Returning the squared mean without filtering"
        # )
        return np.square(np.mean(rms_values))

    return np.square(np.mean(filtered_rms))


def calculate_S(ecg_signal, qrs_inds, fs):
    p2p_amps = []
    buffer = int((50 / 1000) * fs)

    if len(qrs_inds) == 0:
        # print("Warning: No qrs indices to calculate S. Returning NaN.")
        return np.nan

    for r_peak in qrs_inds:
        start = max(0, r_peak - buffer)
        end = min(len(ecg_signal), r_peak + buffer)
        qrs_segment = ecg_signal[start:end]

        if len(qrs_segment) > 0:
            p2p_amps.append(np.max(qrs_segment) - np.min(qrs_segment))

    if len(p2p_amps) < 3:
        # print(
        #     "Warning: Not enough p2p amps for filtering. Returning S calculated from p2p amps without filtering."
        # )
        return (np.mean(p2p_amps) ** 2) / 8

    lower_bound = np.percentile(p2p_amps, 5)
    upper_bound = np.percentile(p2p_amps, 95)
    filtered_amps = [amp for amp in p2p_amps if lower_bound <= amp <= upper_bound]

    if not filtered_amps:
        # print(
        #     "Warning: Filtering removed all p2p amps. Returning S calculated from p2p amps without filtering."
        # )
        return (np.mean(p2p_amps) ** 2) / 8

    return (np.mean(filtered_amps) ** 2) / 8


def estimate_snr(noisy_signal, fs, wavelet="db6", decomp_level=11):
    estimated_clean_signal, estimated_noise = wavelet_denoise(
        data=noisy_signal, wavelet=wavelet, decomp_level=decomp_level
    )

    qrs_inds = processing.xqrs_detect(sig=estimated_clean_signal, fs=fs, verbose=False)

    S_estimated = calculate_S(estimated_clean_signal, qrs_inds, fs)
    N_estimated = calculate_N(estimated_noise, fs)

    if N_estimated == 0:
        # print("WARNING: Noise power 0 --> Division by zero! Returning inf.")
        return np.inf

    snr_db = 10 * np.log10(S_estimated / N_estimated)

    return snr_db


def adjust_noise_frequency(noise_signal, fs_noise, fs_ecg):
    if fs_noise > 1.1 * fs_ecg or fs_ecg > 1.1 * fs_noise:
        resampled_noise_channels = [
            processing.resample_sig(noise_signal[:, i], fs=fs_noise, fs_target=fs_ecg)[
                0
            ]
            for i in range(noise_signal.shape[1])
        ]
        return np.column_stack(resampled_noise_channels)
    return noise_signal


def add_nst_noise(clean_ecg, noise_signal, fs_ecg, fs_noise, target_snr):
    noise_signal = adjust_noise_frequency(noise_signal, fs_noise, fs_ecg)

    num_samples, num_channels = clean_ecg.shape
    if len(noise_signal) < num_samples:
        repeats = int(np.ceil(num_samples / len(noise_signal)))
        noise_signal = np.tile(noise_signal, (repeats, 1))

    latest_start_index = len(noise_signal) - num_samples
    random_start_index = np.random.randint(0, latest_start_index + 1)
    noise_segment = noise_signal[
        random_start_index : random_start_index + num_samples, :
    ]

    qrs_channel_index = (
        1 if num_channels > 1 else 0
    )  # lead II to detect QRS complexes easier
    qrs_inds = processing.xqrs_detect(
        sig=clean_ecg[:, qrs_channel_index], fs=fs_ecg, verbose=False
    )

    if len(qrs_inds) < 3:
        # print(
        #     "Warning: Fewer than 3 QRS complexes found. Cannot reliably calculate SNR. Returning clean ECG."
        # )
        return clean_ecg, False

    signal_powers = []
    for i in range(num_channels):
        signal_powers.append(calculate_S(clean_ecg[:, i], qrs_inds, fs_ecg))

    noise_channels = noise_segment.shape[1]
    noise_powers = []
    for i in range(num_channels):
        noise_powers.append(calculate_N(noise_segment[:, i % noise_channels], fs_ecg))

    scaling_factors = []
    for i in range(num_channels):
        S = signal_powers[i]
        N = noise_powers[i]

        if S == 0 or N == 0:
            a = 0.0
        else:
            snr_linear = 10 ** (
                target_snr / 10
            )  # Rearranged formula from nst tool: a = sqrt(S / (N * 10^(SNR/10)))
            a = np.sqrt(S / (N * snr_linear))
        scaling_factors.append(a)

    total_duration_s = num_samples / fs_ecg
    noise_duration_s = random.uniform(1.0, total_duration_s)
    latest_start_s = total_duration_s - noise_duration_s
    start_s = random.uniform(0.0, latest_start_s)
    end_s = start_s + noise_duration_s

    start_sample = int(start_s * fs_ecg)
    end_sample = int(end_s * fs_ecg)

    scaling_factors = np.array(scaling_factors)
    noise_channel_ids = np.arange(num_channels) % noise_channels
    scaled_noise = noise_segment[:, noise_channel_ids] * scaling_factors

    noisy_ecg = np.copy(clean_ecg)

    unadjusted_noisy_segment = (
        clean_ecg[start_sample:end_sample, :] + scaled_noise[start_sample:end_sample, :]
    )

    if start_sample > 0:
        previous_sample = clean_ecg[start_sample - 1, :]
        first_noisy_sample = unadjusted_noisy_segment[0, :]
        offset_noise_on = first_noisy_sample - previous_sample
    else:
        offset_noise_on = np.zeros(num_channels)

    adjusted_noisy_segment = unadjusted_noisy_segment - offset_noise_on
    noisy_ecg[start_sample:end_sample, :] = adjusted_noisy_segment

    if end_sample < num_samples:
        last_noisy_sample = adjusted_noisy_segment[-1, :]
        first_post_noise_clean_sample = clean_ecg[end_sample, :]
        offset_noise_off = first_post_noise_clean_sample - last_noisy_sample
        noisy_ecg[end_sample:, :] = clean_ecg[end_sample:, :] - offset_noise_off

    return noisy_ecg, True


def main():
    x_test_noisy = np.load(
        "data/brno-university-of-technology-ecg-quality-database/x_test_10snoise.npy"
    )

    snrs = []
    for i in tqdm(range(x_test_noisy.shape[0])):
        estimated_snr = estimate_snr(np.squeeze(x_test_noisy[i]), fs=1000)
        if np.isfinite(estimated_snr):
            snrs.append(estimated_snr)

    np.save(
        "data/brno-university-of-technology-ecg-quality-database/snr_values.npy",
        snrs,
        allow_pickle=False,
    )

    CONDITIONS = ["HYP", "CD", "STTC", "MI"]
    DATASETS = ["val", "test"]
    VERSIONS = ["ood", "non_ood"]

    label_positions = {"HYP": 1, "CD": 2, "STTC": 3, "MI": 4}

    noise_record = wfdb.rdrecord("...") # path to the mit bih noise stress test database that contains the "em" files
    noise_signal = noise_record.p_signal
    estimated_snrs = np.load(
        "data/brno-university-of-technology-ecg-quality-database/snr_values.npy"
    )

    fs_ecg = 500
    fs_noise = 360

    for CONDITION in CONDITIONS:
        for DATASET in DATASETS:
            x_ecg = np.load(f"data/ptb_xl/x_{DATASET}.npy")
            y_alllabels = np.load(f"data/ptb_xl/y_{DATASET}_all_labels.npy")
            y_anomaly_labels = np.load(f"data/ptb_xl/y_{DATASET}_{CONDITION}.npy")
            pos_idx = np.where(y_anomaly_labels == 1)[0]
            neg_idx = np.where(y_anomaly_labels == 0)[0]
            for VERSION in VERSIONS:
                noisy_ecg_batch = []
                idx_record = []
                if VERSION == "ood":
                    while len(noisy_ecg_batch) < int(len(pos_idx) * 0.165):
                        # print(len(idx_record))
                        rnd_idx = np.random.choice(pos_idx, replace=False)
                        if rnd_idx in idx_record:
                            continue
                        single_clean_ecg = x_ecg[rnd_idx]
                        target_snr = np.random.choice(estimated_snrs)
                        single_noisy_ecg, success = add_nst_noise(
                            clean_ecg=single_clean_ecg,
                            noise_signal=noise_signal,
                            fs_ecg=fs_ecg,
                            fs_noise=fs_noise,
                            target_snr=target_snr,
                        )
                        if success:
                            noisy_ecg_batch.append(single_noisy_ecg)
                            idx_record.append(rnd_idx)

                    final_noisy_batch = np.array(noisy_ecg_batch)
                    remaining_val_pos_idx = np.setdiff1d(pos_idx, idx_record)
                    final_remaining_batch = x_ecg[remaining_val_pos_idx]
                    np.save(
                        f"data/ptb_xl/x_{DATASET}_{VERSION}_noisy_{CONDITION}.npy",
                        final_noisy_batch,
                        allow_pickle=False,
                    )
                    np.save(
                        f"data/ptb_xl/x_{DATASET}_{VERSION}_clean_{CONDITION}.npy",
                        final_remaining_batch,
                        allow_pickle=False,
                    )

                if VERSION == "non_ood":
                    while len(noisy_ecg_batch) < int(len(neg_idx) * 0.165):
                        # print(len(idx_record))
                        rnd_idx = np.random.choice(neg_idx, replace=False)
                        if rnd_idx in idx_record:
                            continue
                        single_clean_ecg = x_ecg[rnd_idx]
                        target_snr = np.random.choice(estimated_snrs)
                        single_noisy_ecg, success = add_nst_noise(
                            clean_ecg=single_clean_ecg,
                            noise_signal=noise_signal,
                            fs_ecg=fs_ecg,
                            fs_noise=fs_noise,
                            target_snr=target_snr,
                        )
                        if success:
                            noisy_ecg_batch.append(single_noisy_ecg)
                            idx_record.append(rnd_idx)

                    final_noisy_batch = np.array(noisy_ecg_batch)
                    remaining_val_neg_idx = np.setdiff1d(neg_idx, idx_record)
                    final_remaining_batch = x_ecg[remaining_val_neg_idx]
                    final_remaining_labels = y_alllabels[remaining_val_neg_idx]
                    final_remaining_labels = np.delete(
                        final_remaining_labels, label_positions[CONDITION], axis=1
                    )

                    np.save(
                        f"data/ptb_xl/x_{DATASET}_{VERSION}_noisy_{CONDITION}.npy",
                        final_noisy_batch,
                        allow_pickle=False,
                    )
                    np.save(
                        f"data/ptb_xl/x_{DATASET}_{VERSION}_clean_{CONDITION}.npy",
                        final_remaining_batch,
                        allow_pickle=False,
                    )
                    np.save(
                        f"data/ptb_xl/y_{DATASET}_{VERSION}_clean_{CONDITION}.npy",
                        final_remaining_labels,
                        allow_pickle=False,
                    )


if __name__ == "__main__":
    main()
