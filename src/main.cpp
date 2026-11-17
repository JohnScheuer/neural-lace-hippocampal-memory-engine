#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

#include "lockfree_spsc_queue.hpp"
#include "latency_profiler.hpp"
#include "volterra_hippocampal_filter.hpp"
#include "theta_gamma_synchronizer.hpp"
#include "hippocampal_safety_sentinel.hpp"
#include "memory_suppression_controller.hpp"
#include "cuda_ca1_stimulation.cuh"

constexpr size_t NUM_CYCLES = 10000;

int main() {
    std::cout << "========================================================\n";
    std::cout << "  NEURAL LACE HIPPOCAMPAL MEMORY ENGINE (DEEP ARCHITECTURE)\n";
    std::cout << "  Author: João Felipe de Souza (github.com/JohnScheuer)\n";
    std::cout << "========================================================\n\n";

    cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "FATAL: No CUDA capable device found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[GPU Target] " << prop.name << " (Compute " << prop.major << "." << prop.minor << ")\n";

    float *h_dg_inputs, *h_ca3_inputs, *h_ca1_targets, *h_biphasic_pulses;
    cudaMallocHost(&h_dg_inputs, DG_CHANNELS * sizeof(float));
    cudaMallocHost(&h_ca3_inputs, CA3_CHANNELS * sizeof(float));
    cudaMallocHost(&h_ca1_targets, CA1_CHANNELS * sizeof(float));
    cudaMallocHost(&h_biphasic_pulses, CA1_CHANNELS * sizeof(float));

    float *d_ca1_targets, *d_biphasic_pulses;
    cudaMalloc(&d_ca1_targets, CA1_CHANNELS * sizeof(float));
    cudaMalloc(&d_biphasic_pulses, CA1_CHANNELS * sizeof(float));

    cudaStream_t cuda_stream;
    cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking);

    VolterraHippocampalFilter volterra_filter;
    ThetaGammaSynchronizer theta_gamma_sync;
    HippocampalSafetySentinel safety_sentinel;
    MemorySuppressionController suppression_controller;
    LatencyProfiler profiler;

    for (size_t i = 0; i < DG_CHANNELS; ++i) {
        h_dg_inputs[i] = (i % 5 == 0) ? 38.0f : 1.8f;
        h_ca3_inputs[i] = (i % 7 == 0) ? 42.0f : 2.2f;
    }

    std::cout << "[Pipeline] Executing 10,000 Cycles with Theta-Gamma Phase Precession @ 1000 Hz...\n";

    double max_recorded_temp_rise = 0.0;

    for (size_t cycle = 0; cycle < NUM_CYCLES; ++cycle) {
        profiler.start();

        MemoryWriteMode mode = (cycle % 2000 < 800) ? MemoryWriteMode::LTP_INSERTION : MemoryWriteMode::NOMINAL_READ;
        float phase_gain = theta_gamma_sync.step_and_get_phase_modulation(mode);

        volterra_filter.decode_dg_ca3_to_ca1(h_dg_inputs, h_ca3_inputs, h_ca1_targets, phase_gain);
        suppression_controller.process_ltp_modulation(h_ca3_inputs, h_ca1_targets, CA1_CHANNELS);

        double temp_rise = 0.0;
        bool safe = safety_sentinel.validate_and_clamp_stimulation(h_ca1_targets, CA1_CHANNELS, temp_rise);
        (void)safe;
        if (temp_rise > max_recorded_temp_rise) max_recorded_temp_rise = temp_rise;

        cudaMemcpyAsync(d_ca1_targets, h_ca1_targets, CA1_CHANNELS * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
        launch_ca1_phased_array_kernel(d_ca1_targets, d_biphasic_pulses, CA1_CHANNELS, cuda_stream);
        cudaMemcpyAsync(h_biphasic_pulses, d_biphasic_pulses, CA1_CHANNELS * sizeof(float), cudaMemcpyDeviceToHost, cuda_stream);

        cudaStreamSynchronize(cuda_stream);

        profiler.stop();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    LatencyStats stats = profiler.compute_stats(500.0);

    std::cout << "\n========================================================\n";
    std::cout << "  DEEP HIPPOCAMPAL BENCHMARK RESULTS (RTX 5060 + Ryzen 5600X)\n";
    std::cout << "========================================================\n";
    std::cout << "  Min Latency        : " << stats.min_us << " us\n";
    std::cout << "  Mean Latency       : " << stats.mean_us << " us\n";
    std::cout << "  P50 Latency        : " << stats.p50_us << " us\n";
    std::cout << "  P90 Latency        : " << stats.p90_us << " us\n";
    std::cout << "  P99 Latency (SLA)  : " << stats.p99_us << " us (Target: < 500.00 us)\n";
    std::cout << "  P99.9 Latency      : " << stats.p99_9_us << " us\n";
    std::cout << "  Max Latency        : " << stats.max_us << " us\n";
    std::cout << "  SLA Violation Rate : " << stats.sla_violation_rate << " %\n";
    std::cout << "  Max Tissue Heating : +" << max_recorded_temp_rise << " deg C (ISO Limit: +0.50 deg C)\n";
    std::cout << "========================================================\n\n";

    profiler.export_csv("hippocampal_latencies.csv");

    std::ofstream json_report("NEURAL_LACE_REPORT.json");
    json_report << "{\n"
                << "  \"gpu\": \"" << prop.name << "\",\n"
                << "  \"cycles\": " << NUM_CYCLES << ",\n"
                << "  \"min_us\": " << stats.min_us << ",\n"
                << "  \"mean_us\": " << stats.mean_us << ",\n"
                << "  \"p50_us\": " << stats.p50_us << ",\n"
                << "  \"p90_us\": " << stats.p90_us << ",\n"
                << "  \"p99_us\": " << stats.p99_us << ",\n"
                << "  \"p99_9_us\": " << stats.p99_9_us << ",\n"
                << "  \"max_us\": " << stats.max_us << ",\n"
                << "  \"sla_violation_rate\": " << stats.sla_violation_rate << ",\n"
                << "  \"max_tissue_temp_rise_c\": " << max_recorded_temp_rise << "\n"
                << "}\n";

    cudaFreeHost(h_dg_inputs);
    cudaFreeHost(h_ca3_inputs);
    cudaFreeHost(h_ca1_targets);
    cudaFreeHost(h_biphasic_pulses);
    cudaFree(d_ca1_targets);
    cudaFree(d_biphasic_pulses);
    cudaStreamDestroy(cuda_stream);

    return 0;
}
