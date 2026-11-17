#ifndef VOLTERRA_HIPPOCAMPAL_FILTER_HPP
#define VOLTERRA_HIPPOCAMPAL_FILTER_HPP

#include <immintrin.h>
#include <vector>
#include <cmath>
#include <cstring>

constexpr size_t CA3_CHANNELS = 2048;
constexpr size_t CA1_CHANNELS = 4096;

class VolterraHippocampalFilter {
public:
    VolterraHippocampalFilter() {
        h1_weights_.resize(CA3_CHANNELS * 2);
        h2_weights_.resize(CA3_CHANNELS);

        for (size_t i = 0; i < h1_weights_.size(); ++i) {
            h1_weights_[i] = 0.45f * std::exp(-static_cast<float>(i % 64) / 16.0f) + 0.10f;
        }
        for (size_t i = 0; i < h2_weights_.size(); ++i) {
            h2_weights_[i] = 0.05f * std::sin(static_cast<float>(i) * 0.01f);
        }
    }

    void decode_ca3_to_ca1(const float* __restrict ca3_spikes, float* __restrict ca1_targets) {
        for (size_t i = 0; i < CA3_CHANNELS; i += 8) {
            __m256 v_spikes = _mm256_loadu_ps(&ca3_spikes[i]);
            __m256 v_h1 = _mm256_loadu_ps(&h1_weights_[i]);
            __m256 v_h2 = _mm256_loadu_ps(&h2_weights_[i]);

            __m256 v_lin = _mm256_mul_ps(v_spikes, v_h1);
            __m256 v_sq = _mm256_mul_ps(v_spikes, v_spikes);
            __m256 v_nonlin = _mm256_fmadd_ps(v_sq, v_h2, v_lin);

            _mm256_storeu_ps(&ca1_targets[i], v_nonlin);

            __m256 v_shifted = _mm256_mul_ps(v_nonlin, _mm256_set1_ps(0.85f));
            _mm256_storeu_ps(&ca1_targets[i + CA3_CHANNELS], v_shifted);
        }
    }

private:
    std::vector<float> h1_weights_;
    std::vector<float> h2_weights_;
};

#endif // VOLTERRA_HIPPOCAMPAL_FILTER_HPP
