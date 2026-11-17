#include "cuda_ca1_stimulation.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void ca1_phased_array_kernel(
    const float* __restrict__ d_ca1_targets,
    float* __restrict__ d_biphasic_pulses,
    int num_channels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_channels) {
        float target_val = d_ca1_targets[idx];

        float left_val = (idx > 0) ? d_ca1_targets[idx - 1] : 0.0f;
        float right_val = (idx < num_channels - 1) ? d_ca1_targets[idx + 1] : 0.0f;

        float inhibited_val = target_val - 0.25f * (left_val + right_val);
        float phase_1 = fmaxf(-25.0f, fminf(25.0f, inhibited_val * 1.8f));
        
        d_biphasic_pulses[idx] = phase_1;
    }
}

extern "C" {
    void launch_ca1_phased_array_kernel(
        const float* d_ca1_targets,
        float* d_biphasic_pulses,
        int num_channels,
        cudaStream_t stream
    ) {
        int threads_per_block = 256;
        int blocks_per_grid = (num_channels + threads_per_block - 1) / threads_per_block;

        ca1_phased_array_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
            d_ca1_targets,
            d_biphasic_pulses,
            num_channels
        );
    }
}
