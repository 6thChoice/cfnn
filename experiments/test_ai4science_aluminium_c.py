from pathlib import Path
import sys

import numpy as np
import pytest


RAW_ROOT = Path(__file__).resolve().parent / "downstream_data" / "raw"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _synthetic_multimode_response():
    frequency = np.linspace(1.0, 120.0, 512)
    response = np.zeros_like(frequency, dtype=np.complex128)
    for center, width, amplitude in (
        (28.0, 1.4, 4.0),
        (76.0, 2.2, 2.7),
        (108.0, 1.8, 3.2),
    ):
        response += amplitude / ((center**2 - frequency**2) + 1j * width * frequency)
    response += 0.002 * (1.0 + 0.2j)
    return frequency, np.column_stack((response.real, response.imag)).astype(np.float32)


def test_full_aluminium_frf_loader_keeps_2049_bin_reference_grid():
    from ai4science_aluminium_c import load_aluminium_frf_full

    curves = load_aluminium_frf_full(RAW_ROOT)

    assert len(curves) == 25
    assert {len(curve.frequency) for curve in curves} == {2049}
    first = curves[0]
    assert first.response.shape == (2049, 2)
    assert first.frequency[0] == pytest.approx(0.0)
    assert first.frequency[-1] == pytest.approx(4266.666666666667)
    assert first.metadata["frf_estimator"] == "H1"
    assert first.metadata["reference_grid"] == "full_2049_bin_h1_frf"
    assert np.all(np.diff(first.frequency) > 0)
    assert np.isfinite(first.response).all()


def test_active_schedules_are_nested_and_reach_requested_budgets():
    from ai4science_aluminium_c import build_active_schedule

    frequency, response = _synthetic_multimode_response()
    schedule = build_active_schedule(
        len(frequency),
        initial_count=12,
        batch_size=4,
        budgets=[24, 48, 96],
        strategy="curvature",
        seed=233,
        frequency=frequency,
        response=response,
    )

    assert sorted(schedule) == [24, 48, 96]
    assert all(len(indices) == budget for budget, indices in schedule.items())
    assert set(schedule[24]).issubset(set(schedule[48]))
    assert set(schedule[48]).issubset(set(schedule[96]))
    assert np.all(np.diff(schedule[96]) > 0)
    assert len(np.unique(schedule[96])) == 96


def test_peak_refine_schedule_targets_predicted_peak_neighborhoods():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 100.0, 501)
    response_complex = (
        0.02
        + 1.0 / ((frequency - 32.0) ** 2 + 0.08)
        + 0.6 / ((frequency - 71.0) ** 2 + 0.12)
    ).astype(np.complex128)
    response = np.column_stack((response_complex.real, response_complex.imag)).astype(np.float32)

    peak_refine = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="peak_refine",
        seed=233,
        frequency=frequency,
        response=response,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def nearest_distance(indices, target_hz):
        return float(np.min(np.abs(frequency[indices] - target_hz)))

    assert set(peak_refine[16]).issubset(set(peak_refine[32]))
    assert nearest_distance(peak_refine[16], 32.0) < nearest_distance(uniform[16], 32.0)
    assert nearest_distance(peak_refine[32], 32.0) <= nearest_distance(uniform[32], 32.0)
    assert nearest_distance(peak_refine[32], 71.0) <= nearest_distance(uniform[32], 71.0)


def test_coverage_peak_refine_balances_peak_targeting_and_global_coverage():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 100.0, 501)
    response_complex = (
        0.02
        + 1.0 / ((frequency - 32.0) ** 2 + 0.08)
        + 0.6 / ((frequency - 71.0) ** 2 + 0.12)
    ).astype(np.complex128)
    response = np.column_stack((response_complex.real, response_complex.imag)).astype(np.float32)

    hybrid = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="coverage_peak_refine",
        seed=233,
        frequency=frequency,
        response=response,
    )
    peak_refine = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="peak_refine",
        seed=233,
        frequency=frequency,
        response=response,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def nearest_distance(indices, target_hz):
        return float(np.min(np.abs(frequency[indices] - target_hz)))

    def largest_gap(indices):
        return float(np.max(np.diff(frequency[indices])))

    assert set(hybrid[16]).issubset(set(hybrid[32]))
    assert nearest_distance(hybrid[32], 32.0) <= nearest_distance(uniform[32], 32.0)
    assert nearest_distance(hybrid[32], 71.0) <= nearest_distance(uniform[32], 71.0)
    assert largest_gap(hybrid[32]) < largest_gap(peak_refine[32])


def test_bootstrap_peak_uncertainty_targets_informative_peak_regions_with_coverage():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 100.0, 501)
    response_complex = (
        0.02
        + 1.0 / ((frequency - 32.0) ** 2 + 0.08)
        + 0.6 / ((frequency - 71.0) ** 2 + 0.12)
    ).astype(np.complex128)
    response = np.column_stack((response_complex.real, response_complex.imag)).astype(np.float32)

    uncertainty = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="bootstrap_peak_uncertainty",
        seed=233,
        frequency=frequency,
        response=response,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def nearest_distance(indices, target_hz):
        return float(np.min(np.abs(frequency[indices] - target_hz)))

    def largest_gap(indices):
        return float(np.max(np.diff(frequency[indices])))

    assert set(uncertainty[16]).issubset(set(uncertainty[32]))
    assert nearest_distance(uncertainty[32], 32.0) <= nearest_distance(uniform[32], 32.0)
    assert nearest_distance(uncertainty[32], 71.0) <= nearest_distance(uniform[32], 71.0)
    assert largest_gap(uncertainty[32]) <= 2.5 * largest_gap(uniform[32])


def test_modal_parameter_uncertainty_targets_candidate_mode_locations_with_coverage():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 100.0, 501)
    response_complex = (
        0.02
        + 1.0 / ((frequency - 32.0) ** 2 + 0.08)
        + 0.6 / ((frequency - 71.0) ** 2 + 0.12)
    ).astype(np.complex128)
    response = np.column_stack((response_complex.real, response_complex.imag)).astype(np.float32)

    modal_uncertainty = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="modal_parameter_uncertainty",
        seed=233,
        frequency=frequency,
        response=response,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[16, 32],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def nearest_distance(indices, target_hz):
        return float(np.min(np.abs(frequency[indices] - target_hz)))

    def largest_gap(indices):
        return float(np.max(np.diff(frequency[indices])))

    assert set(modal_uncertainty[16]).issubset(set(modal_uncertainty[32]))
    assert nearest_distance(modal_uncertainty[32], 32.0) <= nearest_distance(uniform[32], 32.0)
    assert nearest_distance(modal_uncertainty[32], 71.0) <= nearest_distance(uniform[32], 71.0)
    assert largest_gap(modal_uncertainty[32]) <= 2.5 * largest_gap(uniform[32])


def test_coverage_modal_uncertainty_combines_mode_refinement_with_stronger_coverage():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 100.0, 501)
    response_complex = (
        0.02
        + 1.0 / ((frequency - 32.0) ** 2 + 0.08)
        + 0.6 / ((frequency - 71.0) ** 2 + 0.12)
        + 0.5 / ((frequency - 88.0) ** 2 + 0.10)
    ).astype(np.complex128)
    response = np.column_stack((response_complex.real, response_complex.imag)).astype(np.float32)

    combined = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[24, 40],
        strategy="coverage_modal_uncertainty",
        seed=233,
        frequency=frequency,
        response=response,
    )
    modal_uncertainty = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[24, 40],
        strategy="modal_parameter_uncertainty",
        seed=233,
        frequency=frequency,
        response=response,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[24, 40],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def nearest_distance(indices, target_hz):
        return float(np.min(np.abs(frequency[indices] - target_hz)))

    def largest_gap(indices):
        return float(np.max(np.diff(frequency[indices])))

    assert set(combined[24]).issubset(set(combined[40]))
    assert nearest_distance(combined[40], 32.0) <= nearest_distance(uniform[40], 32.0)
    assert nearest_distance(combined[40], 71.0) <= nearest_distance(uniform[40], 71.0)
    assert nearest_distance(combined[40], 88.0) <= nearest_distance(uniform[40], 88.0)
    assert largest_gap(combined[40]) < largest_gap(modal_uncertainty[40])


def test_catalogue_band_coverage_forces_local_mode_band_observations():
    from ai4science_aluminium_c import build_active_schedule

    frequency, response = _synthetic_multimode_response()
    centres = [28.0, 76.0, 108.0]
    schedule = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[24, 40],
        strategy="catalogue_band_coverage",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_frequencies_hz=centres,
        catalogue_band_half_width_hz=4.0,
        catalogue_min_band_points=5,
    )
    uniform = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[24, 40],
        strategy="uniform",
        seed=233,
        frequency=frequency,
        response=response,
    )

    def band_count(indices, centre):
        return int(np.sum(np.abs(frequency[indices] - centre) <= 4.0))

    assert set(schedule[24]).issubset(set(schedule[40]))
    assert all(band_count(schedule[40], centre) >= 5 for centre in centres)
    assert sum(band_count(schedule[40], centre) for centre in centres) > sum(
        band_count(uniform[40], centre) for centre in centres
    )


def test_catalogue_band_informative_keeps_densifying_mode_bands_after_minimum_coverage():
    from ai4science_aluminium_c import build_active_schedule

    frequency, response = _synthetic_multimode_response()
    centres = [28.0, 76.0, 108.0]
    informative = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[40, 64],
        strategy="catalogue_band_informative",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_frequencies_hz=centres,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
    )
    minimum_only = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=4,
        budgets=[40, 64],
        strategy="catalogue_band_coverage",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_frequencies_hz=centres,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
    )

    def band_indices(indices, centre):
        return np.asarray(indices)[np.abs(frequency[indices] - centre) <= 6.0]

    def largest_band_gap(indices, centre):
        local = band_indices(indices, centre)
        return float(np.max(np.diff(frequency[local]))) if len(local) >= 2 else float("inf")

    assert set(informative[40]).issubset(set(informative[64]))
    assert all(len(band_indices(informative[64], centre)) >= 8 for centre in centres)
    assert sum(len(band_indices(informative[64], centre)) for centre in centres) > sum(
        len(band_indices(minimum_only[64], centre)) for centre in centres
    )
    assert max(largest_band_gap(informative[64], centre) for centre in centres) < max(
        largest_band_gap(minimum_only[64], centre) for centre in centres
    )


def test_catalogue_band_half_power_online_does_not_use_unobserved_response_values():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 120.0, 601)
    response_a = np.ones_like(frequency, dtype=np.complex128)
    response_b = response_a.copy()
    initial = np.unique(np.linspace(0, len(frequency) - 1, 9, dtype=np.int64))
    unobserved = np.setdiff1d(np.arange(len(frequency), dtype=np.int64), initial)
    response_b[unobserved] += 50.0 / (1.0 + ((frequency[unobserved] - 76.0) / 1.5) ** 2)
    response_a = np.column_stack((response_a.real, response_a.imag)).astype(np.float32)
    response_b = np.column_stack((response_b.real, response_b.imag)).astype(np.float32)

    schedule_a = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="catalogue_band_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_a,
        catalogue_frequencies_hz=[32.0, 76.0],
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
    )
    schedule_b = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="catalogue_band_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_b,
        catalogue_frequencies_hz=[32.0, 76.0],
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
    )

    assert np.asarray(schedule_a[13]).tolist() == np.asarray(schedule_b[13]).tolist()


def test_observed_peak_half_power_online_discovers_mode_bands_without_catalogue():
    from ai4science_aluminium_c import build_active_schedule

    frequency, response = _synthetic_multimode_response()
    schedule = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[64],
        strategy="observed_peak_half_power_online",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )

    observed = schedule[64]
    assert len(observed) == 64
    assert all(
        np.sum(np.abs(frequency[observed] - centre) <= 6.0) >= 5
        for centre in [28.0, 76.0, 108.0]
    )


def test_observed_peak_half_power_online_does_not_use_unobserved_response_values():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 120.0, 601)
    response_a = np.ones_like(frequency, dtype=np.complex128)
    response_b = response_a.copy()
    initial = np.unique(np.linspace(0, len(frequency) - 1, 9, dtype=np.int64))
    unobserved = np.setdiff1d(np.arange(len(frequency), dtype=np.int64), initial)
    response_b[unobserved] += 50.0 / (1.0 + ((frequency[unobserved] - 76.0) / 1.5) ** 2)
    response_a = np.column_stack((response_a.real, response_a.imag)).astype(np.float32)
    response_b = np.column_stack((response_b.real, response_b.imag)).astype(np.float32)

    schedule_a = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="observed_peak_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_a,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )
    schedule_b = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="observed_peak_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_b,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )

    assert np.asarray(schedule_a[13]).tolist() == np.asarray(schedule_b[13]).tolist()


def test_observed_stability_ranked_half_power_online_does_not_use_unobserved_response_values():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 120.0, 601)
    response_a = np.ones_like(frequency, dtype=np.complex128)
    response_b = response_a.copy()
    initial = np.unique(np.linspace(0, len(frequency) - 1, 9, dtype=np.int64))
    unobserved = np.setdiff1d(np.arange(len(frequency), dtype=np.int64), initial)
    response_b[unobserved] += 80.0 / (1.0 + ((frequency[unobserved] - 76.0) / 1.5) ** 2)
    response_a = np.column_stack((response_a.real, response_a.imag)).astype(np.float32)
    response_b = np.column_stack((response_b.real, response_b.imag)).astype(np.float32)

    schedule_a = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="observed_stability_ranked_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_a,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )
    schedule_b = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13],
        strategy="observed_stability_ranked_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_b,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )

    assert np.asarray(schedule_a[13]).tolist() == np.asarray(schedule_b[13]).tolist()


def test_observed_stability_ranked_half_power_online_improves_real_candidate_coverage_without_false_growth():
    from ai4science_aluminium_c import (
        _match_frequency_candidates,
        _observed_peak_centres_from_observed,
        build_active_schedule,
        extract_modal_events,
        load_aluminium_frf_full,
    )

    curves = {curve.curve_id: curve for curve in load_aluminium_frf_full(RAW_ROOT)}
    curve = curves["point:03"]
    truth_centres = [
        float(mode["frequency_hz"])
        for mode in extract_modal_events(
            curve.frequency,
            curve.response,
            max_modes=5,
            min_frequency_hz=5.0,
        )["modes"][:5]
    ]
    default_observed = build_active_schedule(
        len(curve.frequency),
        initial_count=24,
        batch_size=8,
        budgets=[96],
        strategy="observed_peak_half_power_online",
        seed=233,
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=5,
    )[96]
    ranked_observed = build_active_schedule(
        len(curve.frequency),
        initial_count=24,
        batch_size=8,
        budgets=[96],
        strategy="observed_stability_ranked_half_power_online",
        seed=233,
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=5,
    )[96]

    default_candidates = _observed_peak_centres_from_observed(
        curve.frequency,
        curve.response,
        default_observed,
        band_half_width_hz=8.0,
        max_modes=5,
    )
    from ai4science_aluminium_c import _observed_stability_ranked_centres_from_observed

    ranked_candidates = _observed_stability_ranked_centres_from_observed(
        curve.frequency,
        curve.response,
        ranked_observed,
        band_half_width_hz=8.0,
        max_modes=5,
    )
    default_matches, _, default_false = _match_frequency_candidates(truth_centres, default_candidates, 8.0)
    ranked_matches, _, ranked_false = _match_frequency_candidates(truth_centres, ranked_candidates, 8.0)

    assert set(ranked_observed).issuperset(set(build_active_schedule(
        len(curve.frequency),
        initial_count=24,
        batch_size=8,
        budgets=[64],
        strategy="observed_stability_ranked_half_power_online",
        seed=233,
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=5,
    )[64]))
    assert len(ranked_matches) > len(default_matches)
    assert len(ranked_false) <= len(default_false)


def test_observed_stratified_budget_centres_protect_high_frequency_real_candidates():
    from ai4science_aluminium_c import (
        _match_frequency_candidates,
        _observed_stratified_budget_candidate_rows_from_observed,
        _observed_stratified_budget_centres_from_observed,
        build_active_schedule,
        extract_modal_events,
        load_aluminium_frf_full,
    )

    curves = {curve.curve_id: curve for curve in load_aluminium_frf_full(RAW_ROOT)}
    curve = curves["point:05"]
    truth_centres = [
        float(mode["frequency_hz"])
        for mode in extract_modal_events(
            curve.frequency,
            curve.response,
            max_modes=5,
            min_frequency_hz=5.0,
        )["modes"][:5]
    ]
    observed = build_active_schedule(
        len(curve.frequency),
        initial_count=24,
        batch_size=8,
        budgets=[96],
        strategy="observed_stratified_budget_half_power_online",
        seed=233,
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=5,
    )[96]

    candidates = _observed_stratified_budget_centres_from_observed(
        curve.frequency,
        curve.response,
        observed,
        band_half_width_hz=8.0,
        max_modes=5,
    )
    rows = _observed_stratified_budget_candidate_rows_from_observed(
        curve.frequency,
        curve.response,
        observed,
        band_half_width_hz=8.0,
        max_modes=5,
    )
    matches, missed, false = _match_frequency_candidates(truth_centres, candidates, 8.0)

    assert len(candidates) <= 5
    assert {row["stratum_id"] for row in rows} & {"low", "mid", "high"} == {"low", "mid", "high"}
    assert any(float(value) >= 240.0 for value in candidates)
    assert any(float(value) >= 240.0 for value in truth_centres)
    assert not any(float(value) >= 240.0 for value in missed)
    assert len(matches) >= 3
    assert len(false) <= 2


def test_observed_stratified_budget_half_power_online_is_nested_and_observed_only():
    from ai4science_aluminium_c import build_active_schedule

    frequency = np.linspace(0.0, 120.0, 601)
    response_a = np.ones_like(frequency, dtype=np.complex128)
    response_b = response_a.copy()
    initial = np.unique(np.linspace(0, len(frequency) - 1, 9, dtype=np.int64))
    unobserved = np.setdiff1d(np.arange(len(frequency), dtype=np.int64), initial)
    response_b[unobserved] += 80.0 / (1.0 + ((frequency[unobserved] - 96.0) / 1.5) ** 2)
    response_a = np.column_stack((response_a.real, response_a.imag)).astype(np.float32)
    response_b = np.column_stack((response_b.real, response_b.imag)).astype(np.float32)

    schedule_a = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13, 17],
        strategy="observed_stratified_budget_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_a,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )
    schedule_b = build_active_schedule(
        len(frequency),
        initial_count=9,
        batch_size=4,
        budgets=[13, 17],
        strategy="observed_stratified_budget_half_power_online",
        seed=233,
        frequency=frequency,
        response=response_b,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )

    assert set(schedule_a[13]).issubset(set(schedule_a[17]))
    assert np.asarray(schedule_a[13]).tolist() == np.asarray(schedule_b[13]).tolist()


def test_observed_peak_stratified_half_power_online_improves_real_low_order_half_power_recovery():
    from ai4science_aluminium_c import (
        build_active_schedule,
        extract_modal_events,
        half_power_recovery_metrics,
        interpolate_response,
        load_aluminium_frf_full,
    )

    curves = {curve.curve_id: curve for curve in load_aluminium_frf_full(RAW_ROOT)}
    curve = curves["point:03"]
    truth_centres = [
        float(mode["frequency_hz"])
        for mode in extract_modal_events(
            curve.frequency,
            curve.response,
            max_modes=5,
            min_frequency_hz=5.0,
        )["modes"][:5]
    ]

    schedule = build_active_schedule(
        len(curve.frequency),
        initial_count=24,
        batch_size=8,
        budgets=[96],
        strategy="observed_peak_stratified_half_power_online",
        seed=233,
        frequency=curve.frequency,
        response=curve.response,
        catalogue_frequencies_hz=None,
        catalogue_band_half_width_hz=8.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=5,
    )

    observed = schedule[96]
    prediction = interpolate_response(
        curve.frequency,
        curve.response,
        observed,
        mode="signed_log_linear",
    )
    metrics = half_power_recovery_metrics(
        curve.frequency,
        curve.response,
        prediction,
        candidate_frequencies_hz=truth_centres,
        band_half_width_hz=8.0,
        frequency_threshold_hz=1.0,
        width_relative_threshold=0.20,
    )

    assert metrics["predicted_ok_mode_count"] >= 5
    assert metrics["passed_mode_count"] >= 4


def test_global_complex_modal_prior_recovers_synthetic_modes_from_sparse_observations():
    from ai4science_aluminium_c import (
        aluminium_response_metrics,
        global_complex_modal_prior_response,
        interpolate_response,
    )

    frequency, response = _synthetic_multimode_response()
    observed = np.unique(np.linspace(0, len(frequency) - 1, 96, dtype=np.int64))

    modal_prediction = global_complex_modal_prior_response(
        frequency,
        response,
        observed,
        max_modes=3,
    )
    interpolation = interpolate_response(
        frequency,
        response,
        observed,
        mode="signed_log_linear",
    )

    modal_metrics = aluminium_response_metrics(frequency, response, modal_prediction)
    interpolation_metrics = aluminium_response_metrics(frequency, response, interpolation)

    assert modal_prediction.shape == response.shape
    assert modal_metrics["status"] == "ok"
    assert modal_metrics["matched_mode_count"] >= 3
    assert modal_metrics["complex_nrmse"] < interpolation_metrics["complex_nrmse"]


def test_global_complex_modal_prior_fit_reports_candidate_centers():
    from ai4science_aluminium_c import global_complex_modal_prior_fit

    frequency, response = _synthetic_multimode_response()
    observed = np.unique(np.linspace(0, len(frequency) - 1, 96, dtype=np.int64))

    fit = global_complex_modal_prior_fit(
        frequency,
        response,
        observed,
        max_modes=3,
    )

    assert fit["prediction"].shape == response.shape
    assert fit["diagnostics"]["status"] == "ok"
    assert fit["diagnostics"]["initial_candidate_frequencies_hz"] == pytest.approx([28.0, 76.0, 108.0], abs=2.0)
    assert fit["diagnostics"]["fitted_candidate_frequencies_hz"] == pytest.approx([28.0, 76.0, 108.0], abs=2.0)
    assert len(fit["diagnostics"]["selected_observed_indices"]) == 3


def test_low_order_global_complex_modal_prior_prefers_low_frequency_candidates():
    from ai4science_aluminium_c import low_order_global_complex_modal_prior_fit

    frequency = np.linspace(1.0, 220.0, 1400)
    response = np.zeros_like(frequency, dtype=np.complex128)
    for center, width, amplitude in (
        (22.0, 1.2, 0.8),
        (46.0, 1.4, 0.9),
        (88.0, 1.6, 1.0),
        (170.0, 1.0, 9.0),
        (205.0, 1.0, 10.0),
    ):
        response += amplitude / ((center**2 - frequency**2) + 1j * width * frequency)
    response = np.column_stack((response.real, response.imag)).astype(np.float32)
    observed = np.arange(len(frequency), dtype=np.int64)

    fit = low_order_global_complex_modal_prior_fit(
        frequency,
        response,
        observed,
        max_modes=3,
    )

    assert fit["diagnostics"]["status"] == "ok"
    assert fit["diagnostics"]["candidate_order"] == "low_order_frequency"
    assert fit["diagnostics"]["initial_candidate_frequencies_hz"] == pytest.approx([22.0, 46.0, 88.0], abs=1.0)


def test_catalogue_aware_global_complex_modal_prior_uses_supplied_mode_centers():
    from ai4science_aluminium_c import (
        aluminium_response_metrics,
        catalogue_aware_global_complex_modal_prior_fit,
    )

    frequency, response = _synthetic_multimode_response()
    observed = np.unique(np.concatenate([
        np.linspace(0, len(frequency) - 1, 96, dtype=np.int64),
        np.searchsorted(frequency, [27.8, 28.0, 76.0, 108.0]),
    ]))

    fit = catalogue_aware_global_complex_modal_prior_fit(
        frequency,
        response,
        observed,
        candidate_frequencies_hz=[28.0, 76.0, 108.0],
        center_window_hz=3.0,
        max_modes=3,
    )
    metrics = aluminium_response_metrics(frequency, response, fit["prediction"])

    assert fit["diagnostics"]["status"] == "ok"
    assert fit["diagnostics"]["candidate_order"] == "catalogue_supplied"
    assert fit["diagnostics"]["initial_candidate_frequencies_hz"] == pytest.approx([28.0, 76.0, 108.0])
    assert fit["diagnostics"]["fitted_candidate_frequencies_hz"] == pytest.approx([28.0, 76.0, 108.0], abs=3.0)
    assert metrics["matched_mode_count"] >= 3


def test_local_catalogue_band_modal_fit_recovers_per_mode_parameters():
    from ai4science_aluminium_c import local_catalogue_band_modal_parameter_fit

    frequency, response = _synthetic_multimode_response()
    observed = np.unique(np.concatenate([
        np.linspace(0, len(frequency) - 1, 96, dtype=np.int64),
        np.searchsorted(frequency, [27.4, 28.0, 28.6, 75.4, 76.0, 76.6, 107.4, 108.0, 108.6]),
    ]))

    fit = local_catalogue_band_modal_parameter_fit(
        frequency,
        response,
        observed,
        candidate_frequencies_hz=[28.0, 76.0, 108.0],
        band_half_width_hz=5.0,
        center_window_hz=2.0,
    )

    assert fit["status"] == "ok"
    assert fit["mode_count"] == 3
    assert fit["passed_mode_count"] >= 3
    assert fit["median_frequency_error_hz"] < 1.0
    assert fit["median_width_relative_error"] < 0.20
    assert [row["status"] for row in fit["mode_rows"]] == ["ok", "ok", "ok"]


def test_half_power_modal_parameters_recover_lorentzian_bandwidth():
    from ai4science_aluminium_c import half_power_modal_parameters

    frequency = np.linspace(80.0, 120.0, 4001)
    center = 100.0
    width = 6.0
    magnitude = 1.0 / np.sqrt(1.0 + ((2.0 * (frequency - center) / width) ** 2))
    response = np.column_stack((magnitude, np.zeros_like(magnitude))).astype(np.float32)

    result = half_power_modal_parameters(
        frequency,
        response,
        candidate_frequencies_hz=[center],
        band_half_width_hz=12.0,
    )

    assert result["status"] == "ok"
    assert result["mode_count"] == 1
    row = result["mode_rows"][0]
    assert row["status"] == "ok"
    assert row["frequency_hz"] == pytest.approx(center, abs=0.02)
    assert row["width_hz"] == pytest.approx(width, rel=0.01)
    assert row["damping_ratio"] == pytest.approx(width / (2.0 * center), rel=0.01)


def test_signed_log_complex_transform_roundtrips_large_dynamic_range_response():
    from ai4science_aluminium_c import inverse_signed_log_complex, signed_log_complex

    response = np.asarray([
        [0.0, 0.0],
        [1.0e-6, -2.0e-6],
        [3.5, -4.25],
        [-1000.0, 2500.0],
    ], dtype=np.float32)

    encoded = signed_log_complex(response)
    decoded = inverse_signed_log_complex(encoded)

    assert encoded.shape == response.shape
    assert np.max(np.abs(encoded)) < np.max(np.abs(response))
    assert decoded == pytest.approx(response, rel=1e-6, abs=1e-6)


def test_interpolation_baseline_preserves_observed_complex_points():
    from ai4science_aluminium_c import interpolate_response

    frequency, response = _synthetic_multimode_response()
    observed_indices = np.asarray([0, 25, 100, 250, 511], dtype=np.int64)

    prediction = interpolate_response(
        frequency,
        response,
        observed_indices,
        mode="signed_log_linear",
    )

    assert prediction.shape == response.shape
    assert prediction[observed_indices] == pytest.approx(
        response[observed_indices], rel=1e-5, abs=1e-5
    )
    assert np.isfinite(prediction).all()


def test_modal_peak_fit_response_recovers_modal_events_at_least_as_well_as_signed_log_interpolation():
    from ai4science_aluminium_c import (
        aluminium_response_metrics,
        interpolate_response,
        modal_peak_fit_response,
    )

    frequency, response = _synthetic_multimode_response()
    observed_indices = np.unique(np.concatenate([
        np.linspace(0, len(frequency) - 1, 32, dtype=np.int64),
        np.searchsorted(frequency, [27.6, 28.0, 28.4, 75.5, 76.0, 76.5, 107.5, 108.0, 108.5]),
    ]))

    signed_log_prediction = interpolate_response(
        frequency,
        response,
        observed_indices,
        mode="signed_log_linear",
    )
    modal_prediction = modal_peak_fit_response(
        frequency,
        response,
        observed_indices,
        max_modes=5,
    )

    signed_log_metrics = aluminium_response_metrics(frequency, response, signed_log_prediction)
    modal_metrics = aluminium_response_metrics(frequency, response, modal_prediction)

    assert modal_prediction.shape == response.shape
    assert np.isfinite(modal_prediction).all()
    assert modal_metrics["matched_mode_count"] >= signed_log_metrics["matched_mode_count"]
    assert modal_metrics["modal_frequency_mae_hz"] is not None


def test_complex_modal_fit_response_recovers_modal_events_at_least_as_well_as_peak_fit():
    from ai4science_aluminium_c import (
        aluminium_response_metrics,
        complex_modal_fit_response,
        modal_peak_fit_response,
    )

    frequency, response = _synthetic_multimode_response()
    observed_indices = np.unique(np.concatenate([
        np.linspace(0, len(frequency) - 1, 28, dtype=np.int64),
        np.searchsorted(frequency, [
            27.3, 27.8, 28.0, 28.2, 28.7,
            75.2, 75.7, 76.0, 76.3, 76.8,
            107.2, 107.7, 108.0, 108.3, 108.8,
        ]),
    ]))

    peak_fit_prediction = modal_peak_fit_response(
        frequency,
        response,
        observed_indices,
        max_modes=5,
    )
    complex_fit_prediction = complex_modal_fit_response(
        frequency,
        response,
        observed_indices,
        max_modes=5,
    )

    peak_fit_metrics = aluminium_response_metrics(frequency, response, peak_fit_prediction)
    complex_fit_metrics = aluminium_response_metrics(frequency, response, complex_fit_prediction)

    assert complex_fit_prediction.shape == response.shape
    assert np.isfinite(complex_fit_prediction).all()
    assert complex_fit_metrics["matched_mode_count"] >= peak_fit_metrics["matched_mode_count"]
    assert complex_fit_metrics["modal_frequency_mae_hz"] is not None


def test_modal_metrics_report_frequency_width_and_peak_window_errors():
    from ai4science_aluminium_c import aluminium_response_metrics

    frequency, truth = _synthetic_multimode_response()
    shifted_frequency = frequency * 1.002
    prediction_complex = np.interp(
        frequency,
        shifted_frequency,
        truth[:, 0],
        left=truth[0, 0],
        right=truth[-1, 0],
    ) + 1j * np.interp(
        frequency,
        shifted_frequency,
        truth[:, 1],
        left=truth[0, 1],
        right=truth[-1, 1],
    )
    prediction = np.column_stack((prediction_complex.real, prediction_complex.imag))

    metrics = aluminium_response_metrics(frequency, truth, prediction)

    assert metrics["status"] == "ok"
    assert metrics["complex_nrmse"] > 0.0
    assert metrics["peak_window_complex_nrmse"] > 0.0
    assert metrics["truth_mode_count"] >= 3
    assert metrics["matched_mode_count"] >= 2
    assert metrics["modal_frequency_mae_hz"] is not None
    assert metrics["modal_width_relative_mae"] is not None
    assert metrics["modal_peak_amplitude_relative_mae"] is not None


def test_half_power_recovery_metrics_compare_truth_and_prediction_parameters():
    from ai4science_aluminium_c import half_power_recovery_metrics

    frequency = np.linspace(80.0, 120.0, 4001)
    truth_center = 100.0
    prediction_center = 101.0
    truth_width = 6.0
    prediction_width = 9.0
    truth_magnitude = 1.0 / np.sqrt(1.0 + ((2.0 * (frequency - truth_center) / truth_width) ** 2))
    prediction_magnitude = 1.0 / np.sqrt(
        1.0 + ((2.0 * (frequency - prediction_center) / prediction_width) ** 2)
    )
    truth = np.column_stack((truth_magnitude, np.zeros_like(truth_magnitude))).astype(np.float32)
    prediction = np.column_stack((prediction_magnitude, np.zeros_like(prediction_magnitude))).astype(np.float32)

    metrics = half_power_recovery_metrics(
        frequency,
        truth,
        prediction,
        candidate_frequencies_hz=[truth_center],
        band_half_width_hz=14.0,
    )

    assert metrics["status"] == "ok"
    assert metrics["truth_ok_mode_count"] == 1
    assert metrics["predicted_ok_mode_count"] == 1
    assert metrics["passed_mode_count"] == 0
    assert metrics["median_frequency_error_hz"] == pytest.approx(1.0, abs=0.02)
    assert metrics["median_width_relative_error"] == pytest.approx(0.5, rel=0.02)


def test_modal_extraction_keeps_first_modes_by_frequency_not_loudest_peaks():
    from ai4science_aluminium_c import extract_modal_events

    frequency = np.linspace(1.0, 220.0, 1200)
    response = np.zeros_like(frequency, dtype=np.complex128)
    for center, width, amplitude in (
        (20.0, 1.0, 1.0),
        (45.0, 1.2, 1.1),
        (90.0, 1.4, 1.2),
        (160.0, 1.0, 8.0),
        (195.0, 1.0, 9.0),
    ):
        response += amplitude / ((center**2 - frequency**2) + 1j * width * frequency)
    events = extract_modal_events(
        frequency,
        np.column_stack((response.real, response.imag)).astype(np.float32),
        max_modes=3,
    )

    extracted = [mode["frequency_hz"] for mode in events["modes"]]
    assert extracted == pytest.approx([20.0, 45.0, 90.0], abs=1.0)


def test_aluminium_truth_modal_catalogue_records_ordered_full_curve_modes():
    from ai4science_aluminium_c import aluminium_truth_modal_catalogue
    from downstream_protocol import MeasuredCurve

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )

    rows = aluminium_truth_modal_catalogue([curve], max_modes=3)

    assert [row["mode_rank"] for row in rows] == [1, 2, 3]
    assert [row["curve_id"] for row in rows] == ["point:99"] * 3
    assert [row["frequency_hz"] for row in rows] == pytest.approx([28.0, 76.0, 108.0], abs=1.0)
    assert all(row["width_hz"] > 0.0 for row in rows)
    assert all(row["prominence_db"] > 0.0 for row in rows)


def test_aluminium_modal_recovery_diagnostics_separates_observation_miss_and_false_modes():
    from ai4science_aluminium_c import aluminium_modal_recovery_diagnostics

    frequency = np.linspace(1.0, 120.0, 512)
    observed_indices = np.asarray([
        int(np.argmin(np.abs(frequency - 28.0))),
        int(np.argmin(np.abs(frequency - 76.0))),
    ])
    records = [
        {
            "curve_id": "point:99",
            "family": "Global complex modal prior",
            "sampling_strategy": "uniform",
            "budget": 2,
            "status": "ok",
            "observed_indices": observed_indices.tolist(),
            "metrics": {
                "status": "ok",
                "modal_matching_tolerance_hz": 3.0,
                "truth_events": {
                    "modes": [
                        {"frequency_hz": 28.0, "width_hz": 4.0, "peak_amplitude": 1.0},
                        {"frequency_hz": 76.0, "width_hz": 6.0, "peak_amplitude": 0.8},
                        {"frequency_hz": 108.0, "width_hz": 5.0, "peak_amplitude": 0.7},
                    ]
                },
                "predicted_events": {
                    "modes": [
                        {"frequency_hz": 28.5, "width_hz": 6.0, "peak_amplitude": 0.9},
                        {"frequency_hz": 116.0, "width_hz": 2.0, "peak_amplitude": 0.6},
                    ]
                },
            },
        }
    ]

    rows = aluminium_modal_recovery_diagnostics(
        records,
        frequency_by_curve={"point:99": frequency},
        required_modes=3,
    )

    row = rows[0]
    assert row["observed_covered_mode_count"] == 2
    assert row["predicted_matched_mode_count"] == 1
    assert row["predicted_missed_mode_count"] == 2
    assert row["false_predicted_mode_count"] == 1
    assert row["failure_stage"] == "sparse_observation_miss"
    assert row["missed_truth_frequencies_hz"] == pytest.approx([76.0, 108.0])
    assert row["false_predicted_frequencies_hz"] == pytest.approx([116.0])
    assert row["median_matched_width_relative_error"] == pytest.approx(0.5)


def test_aluminium_modal_recovery_diagnostics_labels_frequency_failure_before_width_failure():
    from ai4science_aluminium_c import aluminium_modal_recovery_diagnostics

    records = [
        {
            "curve_id": "point:99",
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "uniform",
            "budget": 192,
            "status": "ok",
            "metrics": {
                "status": "ok",
                "modal_matching_tolerance_hz": 8.0,
                "truth_events": {
                    "modes": [
                        {"frequency_hz": 30.0, "width_hz": 4.0, "peak_amplitude": 1.0},
                    ]
                },
                "predicted_events": {
                    "modes": [
                        {"frequency_hz": 36.0, "width_hz": 8.0, "peak_amplitude": 0.9},
                    ]
                },
            },
        }
    ]

    row = aluminium_modal_recovery_diagnostics(records, required_modes=1)[0]

    assert row["predicted_matched_mode_count"] == 1
    assert row["median_matched_frequency_error_hz"] == pytest.approx(6.0)
    assert row["median_matched_width_relative_error"] == pytest.approx(1.0)
    assert row["failure_stage"] == "frequency_estimation_failure"


def test_summarize_aluminium_modal_recovery_diagnostics_reports_stage_counts():
    from ai4science_aluminium_c import summarize_aluminium_modal_recovery_diagnostics

    rows = [
        {
            "family": "Global complex modal prior",
            "sampling_strategy": "uniform",
            "budget": 192,
            "failure_stage": "sparse_observation_miss",
            "observed_covered_mode_fraction": 0.4,
            "matched_mode_fraction": 0.2,
            "predicted_missed_mode_count": 4,
            "false_predicted_mode_count": 1,
            "median_matched_width_relative_error": 0.5,
        },
        {
            "family": "Global complex modal prior",
            "sampling_strategy": "uniform",
            "budget": 192,
            "failure_stage": "prediction_mode_miss",
            "observed_covered_mode_fraction": 1.0,
            "matched_mode_fraction": 0.6,
            "predicted_missed_mode_count": 2,
            "false_predicted_mode_count": 0,
            "median_matched_width_relative_error": 0.2,
        },
    ]

    summary = summarize_aluminium_modal_recovery_diagnostics(rows)

    row = summary[0]
    assert row["stage_counts"] == {"prediction_mode_miss": 1, "sparse_observation_miss": 1}
    assert row["dominant_failure_stage"] == "prediction_mode_miss"
    assert row["median_observed_covered_mode_fraction"] == pytest.approx(0.7)
    assert row["median_predicted_missed_mode_count"] == pytest.approx(3.0)
    assert row["median_false_predicted_mode_count"] == pytest.approx(0.5)


def test_aluminium_global_modal_candidate_audit_compares_initial_and_fitted_centers():
    from ai4science_aluminium_c import aluminium_global_modal_candidate_audit

    truth_rows = [
        {"curve_id": "point:99", "mode_rank": 1, "frequency_hz": 28.0},
        {"curve_id": "point:99", "mode_rank": 2, "frequency_hz": 76.0},
        {"curve_id": "point:99", "mode_rank": 3, "frequency_hz": 108.0},
    ]
    records = [
        {
            "curve_id": "point:99",
            "family": "Catalogue-aware global modal prior",
            "sampling_strategy": "uniform",
            "budget": 96,
            "status": "ok",
            "model_metadata": {
                "fit_diagnostics": {
                    "status": "ok",
                    "initial_candidate_frequencies_hz": [27.8, 76.2, 108.1],
                    "fitted_candidate_frequencies_hz": [28.1, 116.0, 150.0],
                }
            },
        }
    ]

    rows = aluminium_global_modal_candidate_audit(records, truth_rows, required_modes=3, tolerance_hz=2.0)

    row = rows[0]
    assert row["family"] == "Catalogue-aware global modal prior"
    assert row["initial_candidate_covered_mode_count"] == 3
    assert row["fitted_candidate_covered_mode_count"] == 1
    assert row["initial_candidate_failure_stage"] == "initial_candidates_cover_truth"
    assert row["fitted_candidate_failure_stage"] == "fitted_center_drift"
    assert row["fitted_missed_truth_frequencies_hz"] == pytest.approx([76.0, 108.0])


def test_aluminium_observed_candidate_diagnostics_reports_candidate_and_crossing_status():
    from ai4science_aluminium_c import (
        aluminium_observed_candidate_diagnostics,
        build_active_schedule,
        half_power_recovery_metrics,
        interpolate_response,
    )
    from downstream_protocol import MeasuredCurve
    from downstream_protocol import MeasuredCurve

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )
    observed = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[64],
        strategy="observed_peak_half_power_online",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )[64]
    prediction = interpolate_response(
        frequency,
        response,
        observed,
        mode="signed_log_linear",
    )
    record = {
        "family": "Signed-log linear interpolation",
        "sampling_strategy": "observed_peak_half_power_online",
        "budget": 64,
        "curve_id": "point:99",
        "status": "ok",
        "observed_indices": observed.tolist(),
        "metrics": {
            "half_power": half_power_recovery_metrics(
                frequency,
                response,
                prediction,
                candidate_frequencies_hz=[28.0, 76.0, 108.0],
                band_half_width_hz=6.0,
                frequency_threshold_hz=1.0,
                width_relative_threshold=0.20,
            )
        },
    }

    rows = aluminium_observed_candidate_diagnostics(
        [record],
        [curve],
        required_modes=3,
        band_half_width_hz=6.0,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_source"] == "observed_peak_prominence"
    assert row["candidate_covered_mode_count"] == 3
    assert row["false_candidate_count"] == 0
    assert row["half_power_passed_mode_count"] == 3
    assert row["truth_aligned_half_power_passed_mode_count"] == 3
    assert row["candidate_failure_stage"] == "pass"


def test_observed_candidate_diagnostics_accepts_stability_ranked_strategy():
    from ai4science_aluminium_c import (
        aluminium_observed_candidate_diagnostics,
        build_active_schedule,
        half_power_recovery_metrics,
        interpolate_response,
    )
    from downstream_protocol import MeasuredCurve

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )
    observed = build_active_schedule(
        len(frequency),
        initial_count=8,
        batch_size=8,
        budgets=[64],
        strategy="observed_stability_ranked_half_power_online",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )[64]
    prediction = interpolate_response(
        frequency,
        response,
        observed,
        mode="signed_log_linear",
    )
    record = {
        "family": "Signed-log linear interpolation",
        "sampling_strategy": "observed_stability_ranked_half_power_online",
        "budget": 64,
        "curve_id": "point:99",
        "status": "ok",
        "observed_indices": observed.tolist(),
        "metrics": {
            "half_power": half_power_recovery_metrics(
                frequency,
                response,
                prediction,
                candidate_frequencies_hz=[28.0, 76.0, 108.0],
                band_half_width_hz=6.0,
                frequency_threshold_hz=1.0,
                width_relative_threshold=0.20,
            )
        },
    }

    rows = aluminium_observed_candidate_diagnostics(
        [record],
        [curve],
        required_modes=3,
        band_half_width_hz=6.0,
    )

    assert len(rows) == 1
    assert rows[0]["candidate_source"] == "observed_stability_ranked"
    assert rows[0]["candidate_covered_mode_count"] == 3


def test_observed_candidate_diagnostics_accepts_stratified_budget_strategy():
    from ai4science_aluminium_c import (
        aluminium_observed_candidate_diagnostics,
        build_active_schedule,
        half_power_recovery_metrics,
        interpolate_response,
    )
    from downstream_protocol import MeasuredCurve

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={},
    )
    observed = build_active_schedule(
        len(frequency),
        initial_count=16,
        batch_size=8,
        budgets=[64],
        strategy="observed_stratified_budget_half_power_online",
        seed=233,
        frequency=frequency,
        response=response,
        catalogue_band_half_width_hz=6.0,
        catalogue_min_band_points=5,
        catalogue_max_modes=3,
    )[64]
    prediction = interpolate_response(
        frequency,
        response,
        observed,
        mode="signed_log_linear",
    )
    record = {
        "family": "Signed-log linear interpolation",
        "sampling_strategy": "observed_stratified_budget_half_power_online",
        "budget": 64,
        "curve_id": "point:99",
        "status": "ok",
        "observed_indices": observed.tolist(),
        "metrics": {
            "half_power": half_power_recovery_metrics(
                frequency,
                response,
                prediction,
                candidate_frequencies_hz=[28.0, 76.0, 108.0],
                band_half_width_hz=6.0,
                frequency_threshold_hz=1.0,
                width_relative_threshold=0.20,
            )
        },
    }

    rows = aluminium_observed_candidate_diagnostics(
        [record],
        [curve],
        required_modes=3,
        band_half_width_hz=6.0,
    )

    assert len(rows) == 1
    assert rows[0]["candidate_source"] == "observed_stratified_budget"
    assert rows[0]["candidate_count"] >= 3


def test_truth_aligned_half_power_counts_ignore_false_candidate_passes():
    from ai4science_aluminium_c import _truth_aligned_half_power_counts

    counts = _truth_aligned_half_power_counts(
        truth_frequencies_hz=[10.0, 20.0, 30.0],
        candidate_frequencies_hz=[10.5, 19.5, 80.0],
        half_power_rows=[
            {"predicted_status": "ok", "passed": True},
            {"predicted_status": "ok", "passed": False},
            {"predicted_status": "ok", "passed": True},
        ],
        tolerance_hz=1.0,
    )

    assert counts["covered_mode_count"] == 2
    assert counts["predicted_ok_mode_count"] == 2
    assert counts["passed_mode_count"] == 1


def test_summarize_aluminium_observed_candidate_diagnostics_reports_dominant_stage():
    from ai4science_aluminium_c import summarize_aluminium_observed_candidate_diagnostics

    rows = [
        {
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "observed_peak_half_power_online",
            "budget": 96,
            "candidate_failure_stage": "candidate_miss",
            "candidate_covered_mode_fraction": 0.4,
            "half_power_passed_mode_count": 2,
            "half_power_predicted_ok_mode_count": 3,
            "truth_aligned_half_power_passed_mode_count": 1,
            "truth_aligned_half_power_predicted_ok_mode_count": 2,
            "false_candidate_count": 2,
            "candidate_conflict_count": 1,
        },
        {
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "observed_peak_half_power_online",
            "budget": 96,
            "candidate_failure_stage": "candidate_miss",
            "candidate_covered_mode_fraction": 0.6,
            "half_power_passed_mode_count": 3,
            "half_power_predicted_ok_mode_count": 4,
            "truth_aligned_half_power_passed_mode_count": 2,
            "truth_aligned_half_power_predicted_ok_mode_count": 3,
            "false_candidate_count": 1,
            "candidate_conflict_count": 0,
        },
        {
            "family": "Complex linear interpolation",
            "sampling_strategy": "observed_peak_half_power_online",
            "budget": 96,
            "candidate_failure_stage": "pass",
            "candidate_covered_mode_fraction": 1.0,
            "half_power_passed_mode_count": 5,
            "half_power_predicted_ok_mode_count": 5,
            "truth_aligned_half_power_passed_mode_count": 5,
            "truth_aligned_half_power_predicted_ok_mode_count": 5,
            "false_candidate_count": 0,
            "candidate_conflict_count": 0,
        },
    ]

    summary = summarize_aluminium_observed_candidate_diagnostics(rows)
    by_family = {row["family"]: row for row in summary}

    assert by_family["Signed-log linear interpolation"]["dominant_candidate_failure_stage"] == "candidate_miss"
    assert by_family["Signed-log linear interpolation"]["median_candidate_covered_mode_fraction"] == pytest.approx(0.5)
    assert by_family["Signed-log linear interpolation"]["median_false_candidate_count"] == pytest.approx(1.5)
    assert by_family["Signed-log linear interpolation"]["median_truth_aligned_half_power_passed_mode_count"] == pytest.approx(1.5)
    assert by_family["Complex linear interpolation"]["dominant_candidate_failure_stage"] == "pass"


def test_aluminium_c_summary_keeps_budget_thresholds_separate():
    from run_ai4science_aluminium_c_pilot import _summarize

    records = [
        {
            "family": "CFNN",
            "sampling_strategy": "uniform",
            "budget": 24,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.20,
                "peak_window_complex_nrmse": 0.30,
                "modal_frequency_mae_hz": 3.0,
                "modal_width_relative_mae": 0.30,
            },
        },
        {
            "family": "CFNN",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.10,
                "peak_window_complex_nrmse": 0.20,
                "modal_frequency_mae_hz": 0.8,
                "modal_width_relative_mae": 0.08,
            },
        },
        {
            "family": "MLP",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.09,
                "peak_window_complex_nrmse": 0.18,
                "modal_frequency_mae_hz": 1.5,
                "modal_width_relative_mae": 0.20,
            },
        },
    ]

    summary = _summarize(
        records,
        frequency_threshold_hz=1.0,
        width_relative_threshold=0.10,
    )

    rows = {
        (row["family"], row["sampling_strategy"], row["budget"]): row
        for row in summary["summary_rows"]
    }
    assert rows[("CFNN", "uniform", 24)]["median_modal_frequency_mae_hz"] == 3.0
    assert rows[("CFNN", "uniform", 48)]["median_modal_frequency_mae_hz"] == 0.8
    thresholds = {
        (row["family"], row["sampling_strategy"]): row
        for row in summary["budget_threshold_rows"]
    }
    assert thresholds[("CFNN", "uniform")]["minimum_budget_meeting_modal_thresholds"] == 48
    assert thresholds[("MLP", "uniform")]["minimum_budget_meeting_modal_thresholds"] is None


def test_modal_peak_fit_baseline_is_registered_and_summarized_separately():
    from run_ai4science_aluminium_c_pilot import INTERPOLATION_BASELINES, _select_families, _summarize

    assert "Modal peak fit interpolation" in INTERPOLATION_BASELINES
    assert "Complex modal fit interpolation" in INTERPOLATION_BASELINES
    assert "Global complex modal prior" in INTERPOLATION_BASELINES
    assert "Low-order global modal prior" in INTERPOLATION_BASELINES
    assert _select_families(include_interpolation_baselines=True, only_interpolation_baselines=True) == list(
        INTERPOLATION_BASELINES
    )

    records = [
        {
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.7,
                "peak_window_complex_nrmse": 0.6,
                "matched_mode_count": 2,
            },
        },
        {
            "family": "Modal peak fit interpolation",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.8,
                "peak_window_complex_nrmse": 0.5,
                "matched_mode_count": 3,
            },
        },
        {
            "family": "Complex modal fit interpolation",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.75,
                "peak_window_complex_nrmse": 0.45,
                "matched_mode_count": 4,
            },
        },
        {
            "family": "Global complex modal prior",
            "sampling_strategy": "uniform",
            "budget": 48,
            "status": "ok",
            "metrics": {
                "complex_nrmse": 0.55,
                "peak_window_complex_nrmse": 0.35,
                "matched_mode_count": 5,
            },
        },
    ]

    rows = {
        (row["family"], row["sampling_strategy"], row["budget"]): row
        for row in _summarize(records)["summary_rows"]
    }

    assert rows[("Signed-log linear interpolation", "uniform", 48)]["median_matched_mode_count"] == 2
    assert rows[("Modal peak fit interpolation", "uniform", 48)]["median_matched_mode_count"] == 3
    assert rows[("Complex modal fit interpolation", "uniform", 48)]["median_matched_mode_count"] == 4
    assert rows[("Global complex modal prior", "uniform", 48)]["median_matched_mode_count"] == 5


def test_target_transform_bundle_marks_metadata_and_preserves_evaluation_target():
    from ai4science_aluminium_c import inverse_signed_log_complex
    from downstream_protocol import MeasuredCurve
    from run_ai4science_aluminium_c_pilot import _make_observed_bundle, _transform_bundle_targets

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )
    bundle = _make_observed_bundle(curve, np.arange(32), seed=233)
    transformed = _transform_bundle_targets(bundle, "signed_log_complex")

    assert transformed.metadata["target_transform"] == "signed_log_complex"
    assert np.max(np.abs(transformed.y_train)) < np.max(np.abs(bundle.y_train))
    assert inverse_signed_log_complex(transformed.y_test) == pytest.approx(bundle.y_test, rel=1e-6, abs=1e-6)


def test_aluminium_runner_schedule_uses_catalogue_band_cli_settings():
    from downstream_protocol import MeasuredCurve
    from run_ai4science_aluminium_c_pilot import _schedule_for_job

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )

    class Args:
        initial_count = 8
        batch_size = 4
        budgets = "40"
        catalogue_band_half_width = 4.0
        catalogue_min_band_points = 5
        catalogue_max_modes = 3

    observed = _schedule_for_job(
        curve,
        {
            "sampling_strategy": "catalogue_band_half_power_online",
            "split_seed": 233,
            "budget": 40,
        },
        Args(),
    )

    assert all(
        np.sum(np.abs(frequency[observed] - centre) <= Args.catalogue_band_half_width) >= Args.catalogue_min_band_points
        for centre in [28.0, 76.0, 108.0]
    )


def test_aluminium_runner_can_disable_truth_catalogue_for_observed_peak_schedule():
    from downstream_protocol import MeasuredCurve
    from run_ai4science_aluminium_c_pilot import _schedule_for_job

    frequency, response = _synthetic_multimode_response()
    curve = MeasuredCurve(
        curve_id="point:99",
        frequency=frequency,
        response=response,
        conditions={"point": 99},
        source_files=("synthetic",),
        metadata={"frf_estimator": "synthetic"},
    )

    class Args:
        initial_count = 8
        batch_size = 8
        budgets = "64"
        catalogue_source = "none"
        catalogue_band_half_width = 6.0
        catalogue_min_band_points = 5
        catalogue_max_modes = 3

    observed = _schedule_for_job(
        curve,
        {
            "sampling_strategy": "observed_peak_half_power_online",
            "split_seed": 233,
            "budget": 64,
        },
        Args(),
    )

    assert len(observed) == 64
    assert all(
        np.sum(np.abs(frequency[observed] - centre) <= Args.catalogue_band_half_width) >= Args.catalogue_min_band_points
        for centre in [28.0, 76.0, 108.0]
    )


def test_aluminium_modal_failure_rows_classify_coverage_frequency_width_and_pass():
    from ai4science_aluminium_c import aluminium_modal_failure_rows

    records = [
        {
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "uniform",
            "budget": 192,
            "curve_id": "point:01",
            "status": "ok",
            "metrics": {
                "truth_mode_count": 5,
                "matched_mode_count": 3,
                "modal_frequency_mae_hz": 0.5,
                "modal_width_relative_mae": 0.05,
            },
        },
        {
            "family": "Complex modal fit interpolation",
            "sampling_strategy": "uniform",
            "budget": 192,
            "curve_id": "point:01",
            "status": "ok",
            "metrics": {
                "truth_mode_count": 5,
                "matched_mode_count": 5,
                "modal_frequency_mae_hz": 2.0,
                "modal_width_relative_mae": 0.05,
            },
        },
        {
            "family": "Modal peak fit interpolation",
            "sampling_strategy": "uniform",
            "budget": 192,
            "curve_id": "point:01",
            "status": "ok",
            "metrics": {
                "truth_mode_count": 5,
                "matched_mode_count": 5,
                "modal_frequency_mae_hz": 0.5,
                "modal_width_relative_mae": 0.25,
            },
        },
        {
            "family": "Oracle",
            "sampling_strategy": "uniform",
            "budget": 192,
            "curve_id": "point:01",
            "status": "ok",
            "metrics": {
                "truth_mode_count": 5,
                "matched_mode_count": 5,
                "modal_frequency_mae_hz": 0.5,
                "modal_width_relative_mae": 0.05,
            },
        },
    ]

    rows = aluminium_modal_failure_rows(
        records,
        required_modes=5,
        frequency_threshold_hz=1.0,
        width_relative_threshold=0.10,
    )

    by_family = {row["family"]: row for row in rows}
    assert by_family["Signed-log linear interpolation"]["failure_reason"] == "coverage_failure"
    assert by_family["Signed-log linear interpolation"]["matched_mode_fraction"] == pytest.approx(0.6)
    assert by_family["Complex modal fit interpolation"]["failure_reason"] == "frequency_failure"
    assert by_family["Modal peak fit interpolation"]["failure_reason"] == "width_failure"
    assert by_family["Oracle"]["failure_reason"] == "pass"


def test_summarize_aluminium_modal_failures_reports_dominant_reason():
    from ai4science_aluminium_c import summarize_aluminium_modal_failures

    rows = [
        {
            "family": "CFNN",
            "sampling_strategy": "uniform",
            "budget": 192,
            "failure_reason": "coverage_failure",
            "matched_mode_fraction": 0.0,
        },
        {
            "family": "CFNN",
            "sampling_strategy": "uniform",
            "budget": 192,
            "failure_reason": "coverage_failure",
            "matched_mode_fraction": 0.2,
        },
        {
            "family": "Signed-log linear interpolation",
            "sampling_strategy": "uniform",
            "budget": 192,
            "failure_reason": "width_failure",
            "matched_mode_fraction": 1.0,
        },
    ]

    summary = summarize_aluminium_modal_failures(rows)
    by_family = {row["family"]: row for row in summary}

    assert by_family["CFNN"]["dominant_failure_reason"] == "coverage_failure"
    assert by_family["CFNN"]["median_matched_mode_fraction"] == pytest.approx(0.1)
    assert by_family["Signed-log linear interpolation"]["dominant_failure_reason"] == "width_failure"


def test_aluminium_active_measurement_value_audit_separates_interpolation_signal_from_cfnn_boundary():
    from ai4science_aluminium_c import audit_aluminium_active_measurement_value

    interpolation_rollup = {
        "summary_rows": [
            {
                "family": "Complex linear interpolation",
                "budget": 64,
                "all_five_pass_count": 5,
                "point_count": 5,
                "median_hp_passed_mode_count": 5.0,
                "median_hp_predicted_ok_mode_count": 5.0,
                "median_peak_window_complex_nrmse": 0.24,
            },
            {
                "family": "Global complex modal prior",
                "budget": 64,
                "all_five_pass_count": 0,
                "point_count": 5,
                "median_hp_passed_mode_count": 0.0,
                "median_hp_predicted_ok_mode_count": 0.0,
                "median_peak_window_complex_nrmse": 1.0,
            },
        ]
    }
    same_schedule = {
        "summary_rows": [
            {
                "family": "CFNN",
                "budget": 64,
                "median_half_power_passed_mode_count": 0.0,
                "median_half_power_predicted_ok_mode_count": 0.0,
                "median_matched_mode_count": 0.0,
            },
            {
                "family": "Complex linear interpolation",
                "budget": 64,
                "median_half_power_passed_mode_count": 5.0,
                "median_half_power_predicted_ok_mode_count": 5.0,
                "median_matched_mode_count": 5.0,
            },
        ]
    }
    candidate_diagnostics = {
        "summary_rows": [
            {
                "family": "Complex linear interpolation",
                "sampling_strategy": "observed_peak_half_power_online",
                "budget": 96,
                "dominant_candidate_failure_stage": "candidate_miss",
                "median_candidate_covered_mode_fraction": 0.8,
                "median_truth_aligned_half_power_passed_mode_count": 3.0,
            }
        ]
    }

    audit = audit_aluminium_active_measurement_value(
        interpolation_rollup,
        same_schedule,
        candidate_diagnostics,
    )
    rows = {row["decision_id"]: row for row in audit["decision_rows"]}

    assert audit["claim_level"] == "measurement_strategy_signal_not_cfnn_advantage"
    assert rows["observed_interpolation_measurement_signal"]["verdict"] == "supports_measurement_efficiency_signal"
    assert rows["observed_interpolation_measurement_signal"]["minimum_budget"] == 64
    assert rows["same_schedule_cfnn_boundary"]["verdict"] == "blocks_cfnn_advantage_claim"
    assert rows["candidate_discovery_boundary"]["dominant_failure_stage"] == "candidate_miss"


def test_aluminium_active_measurement_value_report_contains_decision_rows():
    from run_ai4science_aluminium_c_active_measurement_value_audit import build_report

    report = build_report({
        "analysis_type": "aluminium_c_active_measurement_value_audit",
        "claim_level": "measurement_strategy_signal_not_cfnn_advantage",
        "decision_rows": [
            {
                "decision_id": "observed_interpolation_measurement_signal",
                "verdict": "supports_measurement_efficiency_signal",
                "minimum_budget": 64,
                "median_passed_modes": 5.0,
            },
            {
                "decision_id": "same_schedule_cfnn_boundary",
                "verdict": "blocks_cfnn_advantage_claim",
                "minimum_budget": None,
                "median_passed_modes": 0.0,
            },
        ],
    })

    assert "Aluminium C Active Measurement Value Audit" in report
    assert "measurement_strategy_signal_not_cfnn_advantage" in report
    assert "observed_interpolation_measurement_signal" in report
    assert "blocks_cfnn_advantage_claim" in report
