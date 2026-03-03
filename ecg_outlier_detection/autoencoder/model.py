# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, filter_array, kernel_array, stride_array, **kwargs):
        super().__init__(**kwargs)

        layers = []
        for i in range(len(filter_array)):
            layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv1D(
                            filter_array[i],
                            kernel_array[i],
                            strides=stride_array[i],
                            padding="same",
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                    ]
                )
            )

        self.layers = tf.keras.Sequential(layers)

    def call(self, x, training=None):
        return self.layers(x, training=training)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, filter_array, kernel_array, stride_array, **kwargs):
        super().__init__(**kwargs)

        layers = []
        for i in range(len(filter_array)):
            layers.append(
                tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv1DTranspose(
                            filter_array[i],
                            kernel_array[i],
                            strides=stride_array[i],
                            padding="same",
                        ),
                        tf.keras.layers.BatchNormalization(),
                        tf.keras.layers.ReLU(),
                    ]
                )
            )

        self.layers = tf.keras.Sequential(layers)

    def call(self, x, training=None):
        return self.layers(x, training=training)


class Autoencoder(tf.keras.Model):
    def __init__(self, filter_array, kernel_array, stride_array, latent_dim, **kwargs):
        super().__init__(**kwargs)

        self.filter_array = filter_array
        self.kernel_array = kernel_array
        self.stride_array = stride_array

        self.encoder = Encoder(filter_array, kernel_array, stride_array)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_latent = tf.keras.layers.Dense(latent_dim)
        self.dense_decoder_input = None
        self.reshape = None
        self.decoder = None

    def build(self, input_shape):
        dummy = tf.keras.Input(shape=input_shape[1:])
        self.shape_before_flatten = self.encoder(dummy).shape[1:]
        units = self.shape_before_flatten[0] * self.shape_before_flatten[1]
        self.dense_decoder_input = tf.keras.layers.Dense(units, activation="relu")
        self.reshape = tf.keras.layers.Reshape(self.shape_before_flatten)

        self.decoder = Decoder(
            list(reversed(self.filter_array))[1:] + [dummy.shape[2]],
            list(reversed(self.kernel_array)),
            list(reversed(self.stride_array)),
        )

        super().build(input_shape)

    def call(self, x, training=None):
        x = self.encoder(x, training=training)
        x = self.flatten(x)
        x = self.dense_latent(x)
        x = self.dense_decoder_input(x)
        x = self.reshape(x)
        x = self.decoder(x, training=training)
        return x


class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, filter_array, kernel_array, stride_array, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.filter_array = filter_array
        self.kernel_array = kernel_array
        self.stride_array = stride_array
        self.latent_dim = latent_dim

        self.encoder = Encoder(filter_array, kernel_array, stride_array)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_mean = tf.keras.layers.Dense(latent_dim)
        self.dense_log_var = tf.keras.layers.Dense(latent_dim, activation="tanh")
        self.sampling = Sampling()

        self.dense_decoder_input = None
        self.reshape = None
        self.decoder = None

        self.z_mean = None
        self.z_log_var = None

    def build(self, input_shape):
        dummy = tf.keras.Input(shape=input_shape[1:])
        self.shape_before_flatten = self.encoder(dummy).shape[1:]
        units = self.shape_before_flatten[0] * self.shape_before_flatten[1]
        self.dense_decoder_input = tf.keras.layers.Dense(units, activation="relu")
        self.reshape = tf.keras.layers.Reshape(self.shape_before_flatten)

        self.decoder = Decoder(
            list(reversed(self.filter_array))[1:] + [dummy.shape[2]],
            list(reversed(self.kernel_array)),
            list(reversed(self.stride_array)),
        )

        super().build(input_shape)

    def call(self, x, training=None):
        x_encoded = self.encoder(x, training=training)
        x_flattened = self.flatten(x_encoded)
        self.z_mean = self.dense_mean(x_flattened)
        self.z_log_var = self.dense_log_var(x_flattened)
        z = self.sampling([self.z_mean, self.z_log_var])
        x = self.dense_decoder_input(z)
        x = self.reshape(x)
        x = self.decoder(x, training=training)
        return x
