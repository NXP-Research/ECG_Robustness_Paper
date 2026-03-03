# Copyright 2025 - 2026 NXP
# SPDX-License-Identifier: MIT

import tensorflow as tf


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, kernel_size, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.stride = stride
        self.kernel_size = kernel_size

        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding="same",
            use_bias=False,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv1D(
            filters=self.num_filters,
            kernel_size=self.kernel_size,
            padding="same",
            use_bias=False,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.shortcut = None
        self.act_final = tf.keras.layers.ReLU()

    def build(self, input_shape):
        if self.stride != 1 or input_shape[-1] != self.num_filters:
            self.shortcut = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=self.num_filters,
                        kernel_size=1,
                        strides=self.stride,
                        padding="same",
                        use_bias=False,
                    ),
                    tf.keras.layers.BatchNormalization(),
                ]
            )
        super().build(input_shape)

    def call(self, x, training=None):
        residual = x
        if self.shortcut:
            residual = self.shortcut(residual, training=training)

        y = self.conv1(x)
        y = self.bn1(y, training=training)
        y = self.act1(y)

        y = self.conv2(y)
        y = self.bn2(y, training=training)

        return self.act_final(residual + y)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "stride": self.stride,
                "kernel_size": self.kernel_size,
            }
        )
        return config


class ResNet(tf.keras.Model):
    def __init__(
        self,
        initial_kernel_size,
        initial_num_filter,
        initial_stride,
        num_filters_list,
        kernel_size_list,
        strides_list,
        latent_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.initial_kernel_size = initial_kernel_size
        self.initial_num_filter = initial_num_filter
        self.initial_stride = initial_stride
        self.num_filters_list = num_filters_list
        self.kernel_size_list = kernel_size_list
        self.strides_list = strides_list
        self.latent_dim = latent_dim

        self.initial_conv = tf.keras.layers.Conv1D(
            filters=self.initial_num_filter,
            kernel_size=self.initial_kernel_size,
            strides=self.initial_stride,
            padding="same",
            use_bias=False,
        )
        self.initial_bn = tf.keras.layers.BatchNormalization()
        self.initial_act = tf.keras.layers.ReLU()

        self.blocks = []
        for i in range(len(self.num_filters_list)):
            self.blocks.append(
                ResNetBlock(
                    self.num_filters_list[i],
                    self.kernel_size_list[i],
                    self.strides_list[i],
                )
            )

        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.head = tf.keras.layers.Dense(latent_dim, use_bias=False)

    def call(self, x, training=None):
        y = self.initial_conv(x)
        y = self.initial_bn(y, training=training)
        y = self.initial_act(y)

        for block in self.blocks:
            y = block(y, training=training)

        pooled_output = self.pooling(y)
        latent_vector = self.head(pooled_output)
        return latent_vector

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "initial_kernel_size": self.initial_kernel_size,
                "initial_num_filter": self.initial_num_filter,
                "initial_stride": self.initial_stride,
                "num_filters_list": self.num_filters_list,
                "kernel_size_list": self.kernel_size_list,
                "strides_list": self.strides_list,
                "latent_dim": self.latent_dim,
            }
        )
        return config
