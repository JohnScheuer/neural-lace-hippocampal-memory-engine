#ifndef HIPPOCAMPAL_SAFETY_SENTINEL_HPP
#define HIPPOCAMPAL_SAFETY_SENTINEL_HPP

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

class HippocampalSafetySentinel {
public:
    static constexpr double MAX_TISSUE_TEMP_RISE_C = 0.50;
    static constexpr size_t WINDOW_SIZE = 100;

    HippocampalSafetySentinel() : window_index_(0), accumulated_energy_(0.0) {
        energy_window_.resize(WINDOW_SIZE, 0.0);
    }

    bool validate_and_clamp_stimulation(float* ca1_pulses, size_t num_channels, double& current_temp_rise_c) {
        double cycle_energy = 0.0;

        for (size_t i = 0; i < num_channels; ++i) {
            float amplitude = std::abs(ca1_pulses[i]);
            if (amplitude > 28.5f) {
                ca1_pulses[i] = (ca1_pulses[i] > 0) ? 28.5f : -28.5f;
                amplitude = 28.5f;
            }
            cycle_energy += amplitude * amplitude * 0.0001;
        }

        accumulated_energy_ -= energy_window_[window_index_];
        energy_window_[window_index_] = cycle_energy;
        accumulated_energy_ += cycle_energy;

        window_index_ = (window_index_ + 1) % WINDOW_SIZE;

        current_temp_rise_c = (accumulated_energy_ / WINDOW_SIZE) * 0.038;

        return (current_temp_rise_c < MAX_TISSUE_TEMP_RISE_C);
    }

private:
    size_t window_index_;
    double accumulated_energy_;
    std::vector<double> energy_window_;
};

#endif // HIPPOCAMPAL_SAFETY_SENTINEL_HPP
