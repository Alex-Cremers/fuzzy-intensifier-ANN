
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("IntensWeighting")
  .Input("x: float32")
  .Input("w: float32")
  .Output("gxw: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
shape_inference::ShapeHandle out_shape;	
    shape_inference::ShapeHandle units_shape;	
    TF_RETURN_IF_ERROR(c->Subshape(c->input(1),-1,&units_shape));	
    TF_RETURN_IF_ERROR(c->Concatenate(c->input(0),units_shape,&out_shape));
    c->set_output(0,out_shape);
    return Status();
  });




// using namespace tensorflow;

class IntensWeightingOp : public OpKernel {
public:
  explicit IntensWeightingOp(OpKernelConstruction* context) : OpKernel(context) {}
  
  void Compute(OpKernelContext* context) override {
    // std::cout << "Checking that we reach point 1\n";
    // Grab the input 
    const Tensor& input_x = context->input(0);
    const Tensor& weights = context->input(1);
    TensorShape input_x_shape = input_x.shape();
    // TensorShape weight_shape = weight_shape.shape();
    TensorShape out_shape = input_x_shape;
    // out_shape.RemoveLastDims(1);
    out_shape.AddDim(weights.dim_size(1));
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output_tensor));
    
    // auto output = output_tensor->template tensor<float, 3>();
    // auto output = output_tensor->flat<float>().data();
    //auto output = output_tensor->flat<float>();
    auto output = output_tensor->tensor<float_t, 3>();
    
    const int batch_size = input_x.dim_size(0);
    // const Eigen::Tensor<float, 2>::Dimensions& d = input_x.dimensions();
    // const int input_size = input_x.dim_size(-1);
    // const int units_n = weights.dim_size(-1);
    const int input_size = input_x.dim_size(input_x.dims()-1);
    const int units_n = weights.dim_size(weights.dims()-1);
    auto x = input_x.tensor<float_t, 2>();
    auto w = weights.tensor<float_t, 2>();
    
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < units_n; ++j) {
          if (x(batch, i) == 0 | w(i, j) == 0) {
            output(batch, i, j) = 0.0;
          } else if (w(i, j) > 0) {
            output(batch, i, j) = pow(x(batch, i), 1.0/w(i, j));
          } else {
            output(batch, i, j) = -pow(x(batch, i), -1.0/w(i, j));
          }
        }
      }
    }
  }
};


REGISTER_KERNEL_BUILDER(Name("IntensWeighting").Device(DEVICE_CPU), IntensWeightingOp);


