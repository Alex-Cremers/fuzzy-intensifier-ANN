import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import numpy as np

@ops.RegisterGradient("IntensWeighting")
def _intens_weighting_grad(op, grad):
  """The gradients for `intens_weighting`.

  Args:
    op: The `intens_weighting` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `intens_weighting` op.

  Returns:
    Gradients with respect to the inputs of `intens_weighting` (x and weights).
  """
  input_x = op.inputs[0]
  input_w = op.inputs[1]
  output_gxw = tf.abs(op.outputs[0])
  shape_x = array_ops.shape(input_x)
  shape_w = array_ops.shape(input_w)
  # Don't propagate nan, as they would almost surely match cases where the gradient is uniformly 0 on an open set.
  grad = tf.where(tf.math.is_nan(grad), 0.0, grad)
  # A lot of conditioning for all the special cases
  # x_grad = tf.clip_by_value(tf.pow(input_x, -1.0), -1e12, 1e12) * tf.reduce_sum(grad * output_gxw * tf.where(input_w==0, input_w, tf.pow(input_w, -1.0)), axis = 2)
  max_active_weights = tf.gather_nd(params = grad*input_w, indices = tf.expand_dims(tf.argmax(tf.where(grad == 0.0, 0.0, tf.abs(input_w)), axis = 2), -1), batch_dims = 2)
  x_grad = tf.where(
    input_x == 0, 
    tf.where(abs(max_active_weights)>1.0, 1e8 * tf.math.sign(max_active_weights), 0.0), # technically, should be exactly 1 when equality
    tf.pow(input_x, -1.0) * tf.reduce_sum(grad * output_gxw * tf.where(input_w==0.0, 0.0, tf.pow(input_w, -1.0)), axis = 2)
  )
  # w_grad = tf.negative(tf.clip_by_value(tf.pow(input_w, -2.0), -1e12, 1e12) * tf.transpose(tf.reduce_sum(tf.transpose(grad * output_gxw) * tf.transpose(tf.clip_by_value(tf.math.log(input_x), -1e12, 1e12)), 2)))
  w_grad = tf.where(
    tf.math.logical_and(input_w == 0.0, tf.expand_dims(tf.math.reduce_max(input_x, 0) == 1.0, -1)),
    tf.transpose(tf.reduce_sum(tf.transpose(grad) * tf.transpose(tf.where(input_x==1.0, 1e8, 0.0)), 2)),
    tf.where(
      input_w != 0.0,
      tf.negative(tf.pow(input_w, -2.0) * tf.transpose(tf.reduce_sum(tf.transpose(grad * output_gxw) * tf.transpose(tf.where(input_x==0, 0.0, tf.math.log(input_x))), 2))),
      0.0
    )
  )
  x_grad = tf.where(tf.math.is_nan(x_grad), 0.0, x_grad)
  w_grad = tf.where(tf.math.is_nan(w_grad), 0.0, w_grad)
  return [x_grad, w_grad]
