import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import numpy as np
from register_intens_weighting_grad import *

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

intens_weighting_module = tf.load_op_library("./intens_weighting.so")
intens_weighting = intens_weighting_module.intens_weighting

# Stupid trick to get values between 0 and 1 inclusive:
x = tf.nn.relu6(tf.random_uniform_initializer(minval=-1.0, maxval=7.0)(shape = [10,4], dtype = "float32"))/6.0
w = tf.random_normal_initializer(stddev = 1.0)(shape = [4, 3], dtype = "float32")

# print(intens_weighting(x, w).numpy())


# Check that the result is correct:
x_arr = x.numpy()[0, :]
w_arr = w.numpy()[:, 0]

gxw_arr = np.sign(w_arr)*pow(x, 1.0/abs(w_arr))
gxw_cpp = intens_weighting(x,w).numpy()[:,:,0]
print(abs(gxw_cpp - gxw_arr).numpy().max())

gxw_arr = np.sign(w.numpy().transpose())*pow(x_arr, abs(1.0/w.numpy().transpose()))
gxw_cpp = intens_weighting(x,w).numpy()[0,:,:]
print(abs(gxw_cpp - gxw_arr.transpose()).max())

x1 = tf.Variable(x)
w1 = tf.Variable(w)

with tf.GradientTape() as tape:
  y1 = intens_weighting(x1, w1)

dy_dx = tape.gradient(y1, x1)
# dy_dx.numpy()

# Check that the gradient wrt x is correct
x_arr = x1.numpy()
w_arr = w1.numpy()
# dy_dx_np = (1.0/x_arr[0,:])*np.tensordot((1.0/w_arr), abs(y1.numpy()[0,:,:]).transpose(), axes=1)
dy_dx_np = (1.0/x_arr[0,:])*np.sum((1.0/w_arr)*abs(y1.numpy()[0,:,:]), axis=1)
print(abs(dy_dx.numpy()[0,:] - np.nan_to_num(dy_dx_np)).max())

with tf.GradientTape() as tape:
  y1 = intens_weighting(x1, w1)

dy_dw = tape.gradient(y1, w1)
# dy_dw.numpy()

# Check that the gradient wrt w is correct
dy_dw_np = (1/w_arr[:,0]**2)*np.nan_to_num(-np.log(x_arr)*abs(y1.numpy()[:,:,0])).sum(axis = 0)
print(abs(dy_dw.numpy()[:,0] - np.nan_to_num(dy_dw_np)).max())

