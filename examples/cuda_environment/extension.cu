#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

// CUDAContext includes cublas, cusparse and cusolver headers. The recipe's
// minimal toolkit intentionally obtains those headers from the framework wheels.
__global__ void add_one_kernel(const float* input, float* output, int64_t count) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = input[index] + 1.0f;
    }
}

at::Tensor add_one(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kFloat, "input must contain float32 values");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    c10::cuda::CUDAGuard guard(input.device());
    auto output = at::empty_like(input);
    const auto count = input.numel();
    if (count != 0) {
        const auto stream = c10::cuda::getCurrentCUDAStream(input.get_device());
        add_one_kernel<<<(count + 255) / 256, 256, 0, stream.stream()>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), count);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

TORCH_LIBRARY(colabcola_cuda_example, library) {
    library.def("add_one(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(colabcola_cuda_example, CUDA, library) {
    library.impl("add_one", &add_one);
}
