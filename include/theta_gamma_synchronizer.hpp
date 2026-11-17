#ifndef THETA_GAMMA_SYNCHRONIZER_HPP
#define THETA_GAMMA_SYNCHRONIZER_HPP

#include <cmath>
#include <cstddef>

enum class MemoryWriteMode {
    LTP_INSERTION, // Long-Term Potentiation (Peak of Theta Wave)
    LTD_ERASURE,   // Long-Term Depression (Trough of Theta Wave)
    NOMINAL_READ
};

class ThetaGammaSynchronizer {
public:
    ThetaGammaSynchronizer() 
        : phase_rad_(0.0), theta_freq_hz_(8.0), gamma_freq_hz_(40.0), dt_sec_(0.001) {}

    // Advance theta/gamma oscillations and calculate dynamic phase-locking scale factor
    float step_and_get_phase_modulation(MemoryWriteMode mode) {
        phase_rad_ += 2.0 * M_PI * theta_freq_hz_ * dt_sec_;
        if (phase_rad_ >= 2.0 * M_PI) phase_rad_ -= 2.0 * M_PI;

        double theta_wave = std::sin(phase_rad_);
        double gamma_wave = std::cos(phase_rad_ * (gamma_freq_hz_ / theta_freq_hz_));

        // Combined Theta-Gamma Phase Precession Envelope
        double envelope = 0.5 * (1.0 + theta_wave) * (0.8 + 0.2 * gamma_wave);

        if (mode == MemoryWriteMode::LTP_INSERTION) {
            return static_cast<float>(envelope * 1.4);
        } else if (mode == MemoryWriteMode::LTD_ERASURE) {
            return static_cast<float>((1.0 - envelope) * 0.2);
        }

        return static_cast<float>(envelope);
    }

    double current_phase_rad() const { return phase_rad_; }

private:
    double phase_rad_;
    double theta_freq_hz_;
    double gamma_freq_hz_;
    double dt_sec_;
};

#endif // THETA_GAMMA_SYNCHRONIZER_HPP
