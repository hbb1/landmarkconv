#include <torch/extension.h>
#include<vector>
#define CHECK_CUDA(x) TORCH_CHECK(!x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
CHECK_CONTIGUOUS(x);   \
CHECK_CUDA(x)

namespace landmarkconv {

#ifdef WITH_CUDA
std::vector<at::Tensor> I1_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I1_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

std::vector<at::Tensor> I2_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I2_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);


std::vector<at::Tensor> I3_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I3_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

std::vector<at::Tensor> I4_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I4_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

std::vector<at::Tensor> I5_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I5_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

std::vector<at::Tensor> I6_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I6_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);


std::vector<at::Tensor> I7_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I7_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

std::vector<at::Tensor> I8_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide);

std::vector<at::Tensor> I8_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
);

#endif 

std::vector<at::Tensor> I1_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I1_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I1_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I1_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

std::vector<at::Tensor> I2_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I2_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I2_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I2_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}


std::vector<at::Tensor> I3_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I3_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I3_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I3_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

std::vector<at::Tensor> I4_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I4_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I4_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I4_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

std::vector<at::Tensor> I5_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I5_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I5_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I5_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

std::vector<at::Tensor> I6_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I6_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I6_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I6_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}


std::vector<at::Tensor> I7_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I7_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I7_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I7_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

std::vector<at::Tensor> I8_pool_forward(
    const at::Tensor & input, 
    const at::Tensor & guide
) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    // CHECK_CUDA(input);
    // CHECK_CUDA(guide);
#ifdef WITH_CUDA
    return I8_pool_forward_laucher(
        input, guide
    );
#else
      AT_ERROR("Not compiled with GPU support");
#endif
}

std::vector<at::Tensor> I8_pool_backward(
    const at::Tensor & input, 
    const at::Tensor & guide, 
    const at::Tensor & output,
    const at::Tensor & maxout,
    const at::Tensor & grad_output

) {
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(guide);
    #ifdef WITH_CUDA
    return I8_pool_backward_laucher(
        input, guide, output, maxout, grad_output
    );
    #else
      AT_ERROR("Not compiled with GPU support");
    #endif
}

}