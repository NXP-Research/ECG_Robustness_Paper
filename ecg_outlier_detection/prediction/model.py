# Copyright 2022 The TensorFlow Authors
# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: Apache-2.0

import config
import numpy as np
import tensorflow as tf


# The following function "fixed_positional_encoding" was adapted from TensorFlow Text
# Source: https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
# Licensed under the Apache License, Version 2.0 (the "License"): https://www.apache.org/licenses/LICENSE-2.0
def fixed_positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, learned=True, max_length=4096):
        super().__init__()
        self.d_model = d_model
        self.learned = learned

        if learned:
            self.learned_pos_encoding = self.add_weight(
                shape=(max_length, d_model),
                initializer="glorot_uniform",
                trainable=True,
            )
        else:
            self.fixed_pos_encoding = fixed_positional_encoding(
                length=max_length, depth=d_model
            )

    def call(self, x):
        length = tf.shape(x)[1]
        if self.learned:
            return x + self.learned_pos_encoding[tf.newaxis, :length, :]
        else:
            return x + self.fixed_pos_encoding[tf.newaxis, :length, :]


# The following class "FeedForward" was adapted from TensorFlow Text
# Source: https://github.com/tensorflow/text/blob/master/docs/tutorials/transformer.ipynb
# Licensed under the Apache License, Version 2.0 (the "License"): https://www.apache.org/licenses/LICENSE-2.0
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="gelu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )

    def call(self, x, training=None):
        return self.seq(x, training=training)


class RevIN(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x, return_stats=False):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        variance = tf.math.reduce_variance(x, axis=1, keepdims=True)
        std = tf.math.sqrt(variance + self.eps)

        x_norm = (x - mean) / std

        if return_stats:
            return x_norm, mean, std
        return x_norm

    def denorm(self, x, mean, std):
        return x * std + mean


class PatchTSTEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        patch_len,
        stride,
        max_seq_len,
        input_feature_dim,
        dropout_rate=0.1,
        use_positional_encoding=True,
        use_causal_mask=False,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        self.use_causal_mask = use_causal_mask

        self.patch_embedding = tf.keras.layers.Dense(d_model)

        self.patch_embedding.build(input_shape=(None, None, input_feature_dim))

        if use_positional_encoding:
            max_patches = (max_seq_len - patch_len) // stride + 1
            self.pos_embedding = PositionalEncoding(
                d_model, learned=True, max_length=max_patches
            )

        self.encoder_layers = [
            (
                tf.keras.layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=d_model // num_heads,
                    dropout=dropout_rate,
                ),
                FeedForward(d_model, dff, dropout_rate),
            )
            for _ in range(num_layers)
        ]

        self.layer_norms_mha = [
            tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)
        ]
        self.layer_norms_ffn = [
            tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def create_patches(self, x):
        patches = tf.signal.frame(
            x,
            frame_length=self.patch_len,
            frame_step=self.stride,
            axis=1,
        )
        return patches

    def call(self, x, training=None):
        patches = self.create_patches(x)

        batch_size = tf.shape(patches)[0]
        num_patches = tf.shape(patches)[1]
        patches_flattened = tf.reshape(patches, [batch_size, num_patches, -1])

        x_embedded = self.patch_embedding(patches_flattened)

        if self.use_positional_encoding:
            x_embedded = self.pos_embedding(x_embedded)

        x_processed = self.dropout(x_embedded, training=training)

        for i, (mha, ffn) in enumerate(self.encoder_layers):
            x_norm = self.layer_norms_mha[i](x_processed)
            attention_output = mha(
                query=x_norm,
                value=x_norm,
                key=x_norm,
                use_causal_mask=self.use_causal_mask,
                training=training,
            )
            x_processed = x_processed + self.dropout(
                attention_output, training=training
            )

            x_norm = self.layer_norms_ffn[i](x_processed)
            ffn_output = ffn(x_norm, training=training)
            x_processed = x_processed + ffn_output

        return x_processed


class PatchTST(tf.keras.Model):
    def __init__(
        self,
        num_layers=3,
        d_model=128,
        num_heads=16,
        dff=256,
        patch_len=16,
        stride=8,
        dropout_rate=0.1,
        use_revin=True,
        use_causal_mask=False,
        channels_together=False,
    ):
        super().__init__()

        self.num_total_channels = config.CHANNELS
        self.patch_len = patch_len
        self.stride = stride
        self.use_revin = use_revin
        self.channels_together = channels_together

        if use_revin:
            self.revin = RevIN()

        if self.channels_together:
            encoder_input_feature_dim = patch_len * self.num_total_channels
            output_projection_dim = encoder_input_feature_dim
        else:
            encoder_input_feature_dim = patch_len
            output_projection_dim = encoder_input_feature_dim

        self.encoder = PatchTSTEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=config.TIME_STEPS,
            input_feature_dim=encoder_input_feature_dim,
            dropout_rate=dropout_rate,
            use_causal_mask=use_causal_mask,
        )

        self.output_projection = tf.keras.layers.Dense(output_projection_dim)

    def call(self, x, training=None):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        if self.use_revin:
            x_norm, mean, std = self.revin(x, return_stats=True)
        else:
            x_norm = x
            mean, std = None, None

        if self.channels_together:
            encoded_patches = self.encoder(x_norm, training=training)
            projected_patches = self.output_projection(encoded_patches)

            num_patches = tf.shape(projected_patches)[1]
            patches_for_recon = tf.reshape(
                projected_patches,
                [batch_size, num_patches, self.patch_len, self.num_total_channels],
            )
            patches_for_recon = tf.transpose(patches_for_recon, [0, 3, 1, 2])
            patches_for_recon = tf.reshape(
                patches_for_recon,
                [batch_size * self.num_total_channels, num_patches, self.patch_len],
            )
        else:
            x_transposed = tf.transpose(x_norm, [0, 2, 1])
            x_reshaped = tf.reshape(
                x_transposed, [batch_size * self.num_total_channels, seq_len, 1]
            )
            encoded_patches = self.encoder(x_reshaped, training=training)
            patches_for_recon = self.output_projection(encoded_patches)

        summed_reconstruction = tf.signal.overlap_and_add(
            patches_for_recon, frame_step=self.stride
        )
        ones_patches = tf.ones_like(patches_for_recon)
        overlap_counts = tf.signal.overlap_and_add(ones_patches, frame_step=self.stride)
        overlap_counts = tf.maximum(overlap_counts, 1.0)
        reconstructed_seqs = summed_reconstruction / overlap_counts
        output_reshaped = tf.reshape(
            reconstructed_seqs, [batch_size, self.num_total_channels, -1]
        )
        output = tf.transpose(output_reshaped, [0, 2, 1])
        output = output[:, :seq_len, :]

        if self.use_revin and mean is not None:
            output = self.revin.denorm(output, mean, std)

        return output
