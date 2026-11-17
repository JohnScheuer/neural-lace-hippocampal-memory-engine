#ifndef CUDA_CA1_STIMULATION_CUH
#define CUDA_CA1_STIMULATION_CUH

#include <cuda_runtime.h>

extern "C" {
    void launch_ca1_phased_array_kernel(
        const float* d_ca1_targets,
        float* d_biphasic_pulses,
        int num_channels,
        cudaStream_t stream
    );
}

#endif // CUDA_CA1_STIMULATION_CUH
