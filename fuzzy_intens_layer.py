import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
# from tensorflow.keras import regularizers
import keras

intens_weighting_module = tf.load_op_library("./intens_weighting.so")
intens_weighting = intens_weighting_module.intens_weighting
from register_intens_weighting_grad import *


class FuzzyIntens(keras.layers.Layer):
    def __init__(self, units=32, kernel_regularizer=None, bias_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5),
            trainable=True,
            regularizer=self.kernel_regularizer
        )
        self.b = self.add_weight(
            shape=(self.units,), 
            initializer=tf.keras.initializers.RandomNormal(mean=0.5, stddev=input_shape[-1]*0.05), 
            trainable=True,
            regularizer=self.bias_regularizer
        )

    def call(self, inputs):
        # return tf.nn.relu(tf.reduce_sum(intens_weighting(inputs, self.w), axis = -2) + self.b)
      return tf.keras.backend.relu(tf.reduce_sum(intens_weighting(inputs, self.w), axis = -2) + self.b, max_value = 1.0)
