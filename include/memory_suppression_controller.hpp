#ifndef MEMORY_SUPPRESSION_CONTROLLER_HPP
#define MEMORY_SUPPRESSION_CONTROLLER_HPP

#include <vector>
#include <numeric>
#include <cmath>

class MemorySuppressionController {
public:
    MemorySuppressionController() : hypersynchrony_threshold_(85.0f), suppression_gain_(1.0f) {}

    void process_ltp_modulation(const float* ca3_inputs, float* ca1_targets, size_t num_channels) {
        float mean_activity = 0.0f;
        for (size_t i = 0; i < 256; ++i) {
            mean_activity += std::abs(ca3_inputs[i]);
        }
        mean_activity /= 256.0f;

        if (mean_activity > hypersynchrony_threshold_) {
            suppression_gain_ = 0.15f; 
        } else {
            suppression_gain_ = 0.95f * suppression_gain_ + 0.05f * 1.0f;
        }

        for (size_t i = 0; i < num_channels; ++i) {
            ca1_targets[i] *= suppression_gain_;
        }
    }

    float current_gain() const { return suppression_gain_; }

private:
    float hypersynchrony_threshold_;
    float suppression_gain_;
};

#endif // MEMORY_SUPPRESSION_CONTROLLER_HPP
