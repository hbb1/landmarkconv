// #include <torch/torch.h>
// name should be different from .cpp file!!!
#include <torch/types.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
    i += blockDim.x * gridDim.x)

// #define THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 128

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void tl_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = 0; j < sw; j++) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i > 0)  x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out = x1;
                if (out < x2) out = x2;
                if (out < x3) out = x3;
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = sigmoid * out + (1-sigmoid) * x1;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void tl_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_x2 = x1;
                auto out_x3 = x1;
                
                if (j > 0) out_x2 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i > 0) out_x3 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto g1 = scalar_t(x1 >= out);
                auto g2 = (1-g1) * scalar_t(out_x2 >= out);
                auto g3 = (1-g1) * (1-g2) * scalar_t(out_x3 >= out);
                
                auto grad_x1 = sigmoid * g1 + (1-sigmoid);
                auto grad_x2 = sigmoid * g2;
                auto grad_x3 = sigmoid * g3;
                auto grad_sigmoid = out - x1;
                // printf("(%i, %i) x1 %f out %f out_x2 %f out_x3 %f g1 %f g2 %f g3 %f gsig %f \n", i, j, x1, out, out_x2, out_x3, g1, g2, g3, grad_sigmoid);
                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_sigmoid * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                  // grad_x3 * *(gradout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
            }
        }
    }   
}


template <typename scalar_t>
__global__ void tr_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i > 0) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                // printf("%f %f %f %f \n", x1, x2, x3, sigmoid);
                // printf("(%i, %i) x1 %f x2 %f x3 %f sigmoid %f \n", i, j, x1, x2, x3, sigmoid);
                // calculate the max values
                auto out = x1;
                if (out < x2) out = x2;
                if (out < x3) out = x3;
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = sigmoid * out + (1-sigmoid) * x1;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void tr_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = 0; j < sw; j++) {
                auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_x2 = x1;
                auto out_x3 = x1;
                
                if (j < sw-1) out_x2 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i > 0) out_x3 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j);
                auto g1 = scalar_t(x1 >= out);
                auto g2 = (1-g1) * scalar_t(out_x2 >= out);
                auto g3 = (1-g1) * (1-g2) * scalar_t(out_x3 >= out);
                
                auto grad_x1 = sigmoid * g1 + (1-sigmoid);
                auto grad_x2 = sigmoid * g2;
                auto grad_x3 = sigmoid * g3;
                auto grad_sigmoid = out - x1;
                // printf("(%i, %i) x1 %f out %f out_x2 %f out_x3 %f g1 %f g2 %f g3 %f gsig %f \n", i, j, x1, out, out_x2, out_x3, g1, g2, g3, grad_sigmoid);
                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_sigmoid * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i-1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}

template <typename scalar_t>
__global__ void bl_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = 0; j < sw; j++) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                if (j > 0) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                // calculate the max values
                auto out = x1;
                if (out < x2) out = x2;
                if (out < x3) out = x3;
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = sigmoid * out + (1-sigmoid) * x1;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void bl_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_x2 = x1;
                auto out_x3 = x1;
                
                if (j > 0) out_x2 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j-1);
                if (i < sh-1) out_x3 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                auto g1 = scalar_t(x1 >= out);
                auto g2 = (1-g1) * scalar_t(out_x2 >= out);
                auto g3 = (1-g1) * (1-g2) * scalar_t(out_x3 >= out);
                
                auto grad_x1 = sigmoid * g1 + (1-sigmoid);
                auto grad_x2 = sigmoid * g2;
                auto grad_x3 = sigmoid * g3;
                auto grad_sigmoid = out - x1;
                // printf("(%i, %i) x1 %f out %f out_x2 %f out_x3 %f g1 %f g2 %f g3 %f gsig %f \n", i, j, x1, out, out_x2, out_x3, g1, g2, g3, grad_sigmoid);
                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_sigmoid * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j > 0) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j-1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}



template <typename scalar_t>
__global__ void br_forward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  scalar_t *max_ptr,
                                  scalar_t *outptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = sh-1; i >= 0; i--) {
            for (int j = sw-1; j >= 0; j--) {
                auto x1 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto x2 = x1;
                auto x3 = x1;
                if (j < sw-1) x2 = *(outptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i < sh-1) x3 = *(outptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                // calculate the max values
                auto out = x1;
                if (out < x2) out = x2;
                if (out < x3) out = x3;
                outptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = sigmoid * out + (1-sigmoid) * x1;
                max_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = out;
            }
        }
    }   
}

template <typename scalar_t>
__global__ void br_backward_kernel(const int nthreads, 
                                  const scalar_t *input_ptr, 
                                  const scalar_t *guide_ptr,
                                  const scalar_t *output_ptr,
                                  const scalar_t *maxout_ptr,
                                  scalar_t *gradout_ptr,
                                  scalar_t *gradin_ptr,
                                  scalar_t *gradguide_ptr,
                                  const int bs, const int ch,
                                  const int sh, const int sw
                                ) {

    CUDA_1D_KERNEL_LOOP(index, nthreads) {
        int dc = index % ch;
        int db = index / ch;
        for (int i = 0; i < sh; i++) {
            for (int j = 0; j < sw; j++) {
                auto x1 = *(input_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out = *(maxout_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto sigmoid = *(guide_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j);
                auto out_x2 = x1;
                auto out_x3 = x1;
                
                if (j < sw-1) out_x2 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + i * sw + j+1);
                if (i < sh-1) out_x3 = *(output_ptr + db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j);
                auto g1 = scalar_t(x1 >= out);
                auto g2 = (1-g1) * scalar_t(out_x2 >= out);
                auto g3 = (1-g1) * (1-g2) * scalar_t(out_x3 >= out);
                
                auto grad_x1 = sigmoid * g1 + (1-sigmoid);
                auto grad_x2 = sigmoid * g2;
                auto grad_x3 = sigmoid * g3;
                auto grad_sigmoid = out - x1;
                // printf("(%i, %i) x1 %f out %f out_x2 %f out_x3 %f g1 %f g2 %f g3 %f gsig %f \n", i, j, x1, out, out_x2, out_x3, g1, g2, g3, grad_sigmoid);
                gradguide_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_sigmoid * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                gradin_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j] = 
                  grad_x1 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                
                if (j < sw-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j+1] += 
                  grad_x2 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
                if (i < sh-1) gradout_ptr[db * ch * sh * sw + dc * sh * sw + (i+1) * sw + j] += 
                  grad_x3 * gradout_ptr[db * ch * sh * sw + dc * sh * sw + i * sw + j];
            }
        }
    }   
}

namespace landmarkconv {

std::vector<at::Tensor> tl_pool_forward_laucher(
    const at::Tensor &input, 
    const at::Tensor &guide) {
    // Ensure CUDA uses the input tensor device.
    at::DeviceGuard guard(input.device());
    AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
    // printf("call by cuda...\n");

    cudaDeviceSynchronize(); // for print
    auto output = input.clone();
    auto maxout = input.clone();
    int bs = input.size(0);
    int ch = input.size(1);
    int sh = input.size(2);
    int sw = input.size(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "tl_pool_forward_laucher", ([&] {
            const scalar_t *input_ptr = input.data_ptr<scalar_t>();
            const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
            scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
            scalar_t *output_ptr = output.data_ptr<scalar_t>();
            tl_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                          0, at::cuda::getCurrentCUDAStream()>>>(
                bs*ch,
                input_ptr,
                guide_ptr,
                max_ptr,
                output_ptr,
                bs, ch, sh, sw
            );
          }
        )
      );

    THCudaCheck(cudaGetLastError());
    return {
        output,
        maxout
    };
}


std::vector<at::Tensor> tl_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print
 
  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "tl_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        tl_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}

std::vector<at::Tensor> tr_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "tr_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          tr_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> tr_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "tr_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        tr_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}





std::vector<at::Tensor> bl_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "bl_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          bl_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> bl_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
  ) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "bl_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        bl_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}



std::vector<at::Tensor> br_pool_forward_laucher(
  const at::Tensor &input, 
  const at::Tensor &guide) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  // printf("call by cuda...\n");

  cudaDeviceSynchronize(); // for print
  auto output = input.clone();
  auto maxout = input.clone();
  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "br_pool_forward_laucher", ([&] {
          const scalar_t *input_ptr = input.data_ptr<scalar_t>();
          const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
          scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
          scalar_t *output_ptr = output.data_ptr<scalar_t>();
          br_forward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                        0, at::cuda::getCurrentCUDAStream()>>>(
              bs*ch,
              input_ptr,
              guide_ptr,
              max_ptr,
              output_ptr,
              bs, ch, sh, sw
          );
        }
      )
    );

  THCudaCheck(cudaGetLastError());
  return {
      output,
      maxout
  };
}


std::vector<at::Tensor> br_pool_backward_laucher(
  const at::Tensor &input,
  const at::Tensor &guide,
  const at::Tensor &output,
  const at::Tensor &maxout,
  const at::Tensor &grad_output
  ) {
  // Ensure CUDA uses the input tensor device.
  at::DeviceGuard guard(input.device());
  AT_ASSERTM(guide.type().is_cuda(), "map must be a CUDA tensor.");
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor.");
  cudaDeviceSynchronize(); // for print

  auto grad_input = at::zeros_like(input);
  auto grad_guide = at::zeros_like(guide);
  auto gradout = grad_output.clone();

  int bs = input.size(0);
  int ch = input.size(1);
  int sh = input.size(2);
  int sw = input.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input.scalar_type(), "br_pool_backward_laucher", ([&] {
        const scalar_t *input_ptr = input.data_ptr<scalar_t>();
        const scalar_t *guide_ptr = guide.data_ptr<scalar_t>();
        const scalar_t *max_ptr = maxout.data_ptr<scalar_t>();
        const scalar_t *output_ptr = output.data_ptr<scalar_t>();
        scalar_t *gradout_ptr = gradout.data_ptr<scalar_t>();
        scalar_t *gradin_ptr = grad_input.data_ptr<scalar_t>();
        scalar_t *gradguide_ptr = grad_guide.data_ptr<scalar_t>();


        br_backward_kernel<scalar_t><<<GET_BLOCKS(bs*ch), THREADS_PER_BLOCK,
                                      0, at::cuda::getCurrentCUDAStream()>>>(
            bs*ch,
            input_ptr,
            guide_ptr,
            output_ptr,
            max_ptr,
            gradout_ptr, 
            gradin_ptr,
            gradguide_ptr,
            bs, ch, sh, sw
        );
      }
    )
  );

  THCudaCheck(cudaGetLastError());
  return {
    grad_input, 
    grad_guide
  };
}

} // namespace landmarkconv2 