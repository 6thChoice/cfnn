import math
import unittest

import numpy as np

from ai4science_fano import (
    fit_complex_fano_curve,
    magnitude_fano_response,
)


class ComplexFanoReferenceTests(unittest.TestCase):
    def test_synthetic_complex_fano_fit_recovers_reference_parameters(self):
        frequency = np.linspace(4.995e9, 4.998e9, 1001)
        magnitude = magnitude_fano_response(
            frequency,
            f0_hz=4.99645e9,
            gamma_hz=1.55e5,
            q=-1.25,
            background=0.017,
            background_slope_per_hz=1.8e-12,
            amplitude=0.010,
        )
        phase = 0.2 * np.arctan((frequency - 4.99645e9) / 1.55e5)
        truth = magnitude * np.exp(1j * phase)

        fit = fit_complex_fano_curve(
            frequency,
            np.column_stack((truth.real, truth.imag)),
            curve_id="synthetic:fano",
            residual_threshold=1e-3,
        )

        self.assertEqual(fit.status, "ok")
        self.assertLess(abs(fit.f0_hz - 4.99645e9), 2.0e3)
        self.assertLess(abs(fit.gamma_hz - 1.55e5) / 1.55e5, 0.03)
        self.assertLess(abs(fit.q - (-1.25)), 0.05)
        self.assertTrue(math.isfinite(fit.quality_factor))
        self.assertGreater(fit.quality_factor, 1.0e4)
        self.assertGreater(fit.magnitude_peak_hz, frequency[0])
        self.assertLess(fit.magnitude_peak_hz, frequency[-1])
        self.assertGreater(fit.magnitude_valley_hz, frequency[0])
        self.assertLess(fit.magnitude_valley_hz, frequency[-1])
        self.assertGreater(fit.phase_transition_hz, frequency[0])
        self.assertLess(fit.phase_transition_hz, frequency[-1])
        self.assertIsNone(fit.relative_complex_rmse)
        self.assertLess(fit.relative_magnitude_rmse, 1e-6)

    def test_flat_curve_is_marked_unidentifiable(self):
        frequency = np.linspace(1.0, 2.0, 101)
        response = np.column_stack((
            np.full_like(frequency, 0.2),
            np.full_like(frequency, -0.1),
        ))

        fit = fit_complex_fano_curve(
            frequency,
            response,
            curve_id="synthetic:flat",
            residual_threshold=0.15,
        )

        self.assertEqual(fit.status, "unidentifiable")
        self.assertEqual(fit.failure_reason, "insufficient_response_structure")


if __name__ == "__main__":
    unittest.main()
