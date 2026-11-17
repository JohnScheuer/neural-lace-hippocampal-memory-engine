#ifndef VOLTERRA_HIPPOCAMPAL_FILTER_HPP
#define VOLTERRA_HIPPOCAMPAL_FILTER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>

constexpr size_t DG_CHANNELS = 2048;
constexpr size_t CA3_CHANNELS = 2048;
constexpr size_t CA1_CHANNELS = 4096;

class VolterraHippocampalFilter {
public:
    VolterraHippocampalFilter() {
        h1_weights_.resize(DG_CHANNELS);
        h2_weights_.resize(CA3_CHANNELS);

        for (size_t i = 0; i < DG_CHANNELS; ++i) {
            h1_weights_[i] = 0.45f * std::exp(-static_cast<float>(i % 64) / 16.0f) + 0.10f;
            h2_weights_[i] = 0.02f * (std::sin(static_cast<float>(i) * 0.02f) + 1.0f); // Non-negative kernel
        }
    }

    void decode_dg_ca3_to_ca1(const float* __restrict dg_spikes, const float* __restrict ca3_spikes, float* __restrict ca1_targets, float phase_gain) {
        __m256 v_phase = _mm256_set1_ps(phase_gain);
        __m256 v_zero = _mm256_setzero_ps();

        for (size_t i = 0; i < DG_CHANNELS; i += 8) {
            __m256 v_dg = _mm256_loadu_ps(&dg_spikes[i]);
            __m256 v_ca3 = _mm256_loadu_ps(&ca3_spikes[i]);
            __m256 v_h1 = _mm256_loadu_ps(&h1_weights_[i]);
            __m256 v_h2 = _mm256_loadu_ps(&h2_weights_[i]);

            __m256 v_lin = _mm256_mul_ps(v_dg, v_h1);
            __m256 v_sq = _mm256_mul_ps(v_ca3, v_ca3);
            __m256 v_nonlin = _mm256_fmadd_ps(v_sq, v_h2, v_lin);

            // Biological ReLU Rectification (No negative firing rates)
            __m256 v_relu = _mm256_max_ps(v_zero, v_nonlin);

            __m256 v_out1 = _mm256_mul_ps(v_relu, v_phase);
            _mm256_storeu_ps(&ca1_targets[i], v_out1);

            __m256 v_out2 = _mm256_mul_ps(v_out1, _mm256_set1_ps(0.82f));
            _mm256_storeu_ps(&ca1_targets[i + DG_CHANNELS], v_out2);
        }
    }

private:
    std::vector<float> h1_weights_;
    std::vector<float> h2_weights_;
};

#endif // VOLTERRA_HIPPOCAMPAL_FILTER_HPP
