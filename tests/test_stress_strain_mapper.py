import importlib.util
import math
from pathlib import Path
import tkinter as tk
import unittest

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "sxrd_stress_strain_mapper_gui_v3.py"
SPEC = importlib.util.spec_from_file_location("stress_strain_mapper", MODULE_PATH)
mapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mapper)


class StressStrainMapperNumericsTest(unittest.TestCase):
    def test_zero_to_first_finite_uses_first_valid_value_and_preserves_nan(self):
        zeroed, offset = mapper.zero_to_first_finite([np.nan, 2.5, 3.0, np.nan, 1.0])

        self.assertAlmostEqual(offset, 2.5)
        np.testing.assert_allclose(zeroed, [np.nan, 0.0, 0.5, np.nan, -1.5], equal_nan=True)

    def test_zero_to_first_finite_returns_nan_offset_without_valid_values(self):
        zeroed, offset = mapper.zero_to_first_finite([np.nan, np.inf, -np.inf])

        self.assertTrue(math.isnan(offset))
        np.testing.assert_allclose(zeroed, [np.nan, np.inf, -np.inf], equal_nan=True)

    def test_strain_alignment_scales_station_11_percent_to_reference_10_percent(self):
        diag = mapper.compute_strain_alignment_diagnostics(
            ref_eps_fraction=[0.0, 0.05, 0.10],
            station_eps_fraction=[0.0, 0.055, 0.11],
        )

        self.assertAlmostEqual(diag["factor"], 0.10 / 0.11)
        self.assertAlmostEqual(diag["reference_max_strain_percent"], 10.0)
        self.assertAlmostEqual(diag["station_max_strain_percent"], 11.0)
        np.testing.assert_allclose(
            mapper.apply_strain_alignment([0.0, 0.055, 0.11], diag),
            [0.0, 0.05, 0.10],
        )

    def test_strain_alignment_scales_station_8_percent_to_reference_10_percent(self):
        diag = mapper.compute_strain_alignment_diagnostics(
            ref_eps_fraction=[0.0, 0.05, 0.10],
            station_eps_fraction=[0.0, 0.04, 0.08],
        )

        self.assertAlmostEqual(diag["factor"], 1.25)
        self.assertAlmostEqual(diag["reference_max_strain_percent"], 10.0)
        self.assertAlmostEqual(diag["station_max_strain_percent"], 8.0)
        np.testing.assert_allclose(
            mapper.apply_strain_alignment([0.0, 0.04, 0.08], diag),
            [0.0, 0.05, 0.10],
        )

    def test_strain_alignment_requires_positive_reference_and_station_strain(self):
        with self.assertRaisesRegex(ValueError, "无法计算应变对齐系数"):
            mapper.compute_strain_alignment_diagnostics(
                ref_eps_fraction=[0.0, 0.10],
                station_eps_fraction=[0.0, np.nan, -0.01],
            )

    def test_both_mode_range_requires_strain_and_stress_inside_reference(self):
        result = mapper.within_reference_ranges(
            eps_fraction=np.array([0.05, 0.25, 0.10, np.nan]),
            stress_mpa=np.array([50.0, 150.0, 250.0, 75.0]),
            ref_eps_fraction=np.array([0.0, 0.2]),
            ref_stress_mpa=np.array([0.0, 200.0]),
        )

        self.assertEqual(result.tolist(), [True, False, False, False])

    def test_duplicate_reference_strain_values_are_averaged_and_reported(self):
        interp, xmin, xmax, method, diagnostics = mapper.make_interpolator_with_diagnostics(
            [0.0, 0.1, 0.1, 0.2],
            [0.0, 90.0, 110.0, 200.0],
            method=mapper.METHOD_LINEAR,
        )

        self.assertEqual((xmin, xmax, method), (0.0, 0.2, "Linear"))
        self.assertEqual(diagnostics["duplicate_groups"], 1)
        self.assertEqual(diagnostics["duplicate_rows"], 2)
        self.assertAlmostEqual(float(interp(np.array([0.1]))[0]), 100.0)

    def test_inverse_plateau_reports_ambiguous_strain_interval(self):
        intervals = mapper.compute_inverse_strain_intervals(
            eps_fraction=[0.0, 0.1, 0.2, 0.3],
            stress_mpa=[0.0, 100.0, 100.0, 200.0],
            query_stress_mpa=[100.0],
            use_pre_peak=True,
        )

        row = intervals.iloc[0]
        self.assertAlmostEqual(row["inverse_strain_min_fraction"], 0.1)
        self.assertAlmostEqual(row["inverse_strain_max_fraction"], 0.2)
        self.assertTrue(bool(row["inverse_mapping_is_ambiguous"]))
        self.assertAlmostEqual(row["inverse_ambiguity_width_percent"], 10.0)

    def test_inverse_monotonic_curve_reports_single_strain_solution(self):
        intervals = mapper.compute_inverse_strain_intervals(
            eps_fraction=[0.0, 0.1, 0.2],
            stress_mpa=[0.0, 100.0, 200.0],
            query_stress_mpa=[100.0],
            use_pre_peak=True,
        )

        row = intervals.iloc[0]
        self.assertAlmostEqual(row["inverse_strain_min_fraction"], 0.1)
        self.assertAlmostEqual(row["inverse_strain_max_fraction"], 0.1)
        self.assertFalse(bool(row["inverse_mapping_is_ambiguous"]))
        self.assertAlmostEqual(row["inverse_ambiguity_width_percent"], 0.0)

    def test_inverse_out_of_range_stress_reports_nan_interval(self):
        intervals = mapper.compute_inverse_strain_intervals(
            eps_fraction=[0.0, 0.1, 0.2],
            stress_mpa=[0.0, 100.0, 200.0],
            query_stress_mpa=[250.0],
            use_pre_peak=True,
        )

        row = intervals.iloc[0]
        self.assertTrue(math.isnan(row["inverse_strain_min_fraction"]))
        self.assertTrue(math.isnan(row["inverse_strain_max_fraction"]))
        self.assertFalse(bool(row["inverse_mapping_is_ambiguous"]))
        self.assertTrue(math.isnan(row["inverse_ambiguity_width_percent"]))


class StressStrainMapperWizardRecommendationTest(unittest.TestCase):
    def test_reference_recommendation_uses_english_column_names(self):
        df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.01, 0.02],
                "stress_MPa": [0.0, 120.0, 240.0],
                "temperature": [25.0, 25.0, 25.0],
            }
        )

        rec = mapper.recommend_reference_setup(df)

        self.assertEqual(rec["strain_col"], "engineering_strain")
        self.assertEqual(rec["stress_col"], "stress_MPa")
        self.assertEqual(rec["strain_unit"], mapper.STRAIN_FRACTION)
        self.assertEqual(rec["stress_unit"], mapper.STRESS_MPA)
        self.assertIn("列名", rec["strain_reason"])

    def test_reference_recommendation_handles_headerless_numeric_columns(self):
        df = pd.DataFrame(
            {
                "col_1": [0.0, 5.0, 10.0],
                "col_2": [0.0, 200.0, 400.0],
            }
        )

        rec = mapper.recommend_reference_setup(df)

        self.assertEqual(rec["strain_col"], "col_1")
        self.assertEqual(rec["stress_col"], "col_2")
        self.assertEqual(rec["strain_unit"], mapper.STRAIN_PERCENT)
        self.assertEqual(rec["stress_unit"], mapper.STRESS_MPA)

    def test_station_recommendation_prefers_named_strain_input_without_id(self):
        df = pd.DataFrame({"strain": [0.0, 0.01, 0.02], "intensity": [10, 11, 12]})

        rec = mapper.recommend_station_setup(df)

        self.assertEqual(rec["mode"], mapper.MODE_STRAIN_ONLY)
        self.assertEqual(rec["strain_col"], "strain")
        self.assertEqual(rec["stress_col"], "")
        self.assertEqual(rec["id_col"], "")
        self.assertEqual(rec["strain_unit"], mapper.STRAIN_FRACTION)

    def test_station_recommendation_detects_id_and_stress_input(self):
        df = pd.DataFrame({"frame": [4, 5, 6], "sigma_MPa": [100.0, 150.0, 200.0]})

        rec = mapper.recommend_station_setup(df)

        self.assertEqual(rec["mode"], mapper.MODE_STRESS_ONLY)
        self.assertEqual(rec["id_col"], "frame")
        self.assertEqual(rec["stress_col"], "sigma_MPa")
        self.assertEqual(rec["strain_col"], "")
        self.assertEqual(rec["stress_unit"], mapper.STRESS_MPA)

    def test_wizard_state_requires_confirmation_before_running(self):
        self.assertEqual(mapper.get_wizard_state(False, False, False, False), "待加载参考")
        self.assertEqual(mapper.get_wizard_state(True, False, False, False), "待加载线站")
        self.assertEqual(mapper.get_wizard_state(True, True, False, False), "待确认推荐")
        self.assertEqual(mapper.get_wizard_state(True, True, True, False), "可运行")
        self.assertEqual(mapper.get_wizard_state(True, True, True, True), "已完成")


class StressStrainMapperGuiRegressionTest(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display is not available: {exc}")
        self.root.geometry("1280x820+80+80")
        self.messages = []
        self._orig_messagebox = (
            mapper.messagebox.showerror,
            mapper.messagebox.showwarning,
            mapper.messagebox.showinfo,
        )
        mapper.messagebox.showerror = self._record_message("error")
        mapper.messagebox.showwarning = self._record_message("warning")
        mapper.messagebox.showinfo = self._record_message("info")
        self.app = mapper.StressStrainMapperApp(self.root)
        self.root.update_idletasks()
        self.root.update()

    def tearDown(self):
        mapper.messagebox.showerror, mapper.messagebox.showwarning, mapper.messagebox.showinfo = self._orig_messagebox
        if hasattr(self, "root"):
            self.root.destroy()

    def _record_message(self, kind):
        def recorder(title, message="", *args, **kwargs):
            self.messages.append((kind, str(title), str(message)))
            return "ok"

        return recorder

    def _configure_strain_mapping(self, station_df):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.01, 0.02, 0.03],
                "stress_MPa": [0.0, 100.0, 210.0, 330.0],
            }
        )
        self.app.station_df = station_df.copy()
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRAIN_ONLY)
        self.app.station_id_col.set("spectrum_id" if "spectrum_id" in station_df.columns else "")
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.recommendation_confirmed.set(True)
        self.app._update_station_mode_hint()
        self.app._update_wizard_state_display()

    def _configure_stress_mapping(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.01, 0.02, 0.03],
                "stress_MPa": [0.0, 100.0, 210.0, 330.0],
            }
        )
        self.app.station_df = pd.DataFrame({"sigma_MPa": [0.0, 100.0, 210.0]})
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRESS_ONLY)
        self.app.station_id_col.set("")
        self.app.station_stress_col.set("sigma_MPa")
        self.app.station_stress_unit.set(mapper.STRESS_MPA)
        self.app.recommendation_confirmed.set(True)
        self.app._update_station_mode_hint()
        self.app._update_wizard_state_display()

    def _configure_both_mapping(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.01, 0.02, 0.03],
                "stress_MPa": [0.0, 100.0, 210.0, 330.0],
            }
        )
        self.app.station_df = pd.DataFrame(
            {
                "frame": [11, 12, 13],
                "strain": [0.0, 0.01, 0.02],
                "sigma_MPa": [0.0, 100.0, 210.0],
            }
        )
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_BOTH)
        self.app.station_id_col.set("frame")
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.station_stress_col.set("sigma_MPa")
        self.app.station_stress_unit.set(mapper.STRESS_MPA)
        self.app.recommendation_confirmed.set(True)
        self.app._update_station_mode_hint()
        self.app._update_wizard_state_display()

    def test_export_dataframe_puts_final_curve_columns_first_for_all_modes(self):
        cases = [
            ("strain only", lambda: self._configure_strain_mapping(pd.DataFrame({"strain": [0.0, 0.01, 0.02]}))),
            ("stress only", self._configure_stress_mapping),
            ("both", self._configure_both_mapping),
        ]

        for _, configure in cases:
            configure()
            self.app.run_mapping()
            self.root.update()
            self.assertFalse([msg for msg in self.messages if msg[0] == "error"])

            export_df = self.app._build_export_dataframe(self.app.result_df)

            self.assertEqual(
                export_df.columns[:3].tolist(),
                ["spectrum_id", "mapped_strain_percent", "mapped_stress_MPa"],
            )
            self.assertEqual(set(export_df.columns), set(self.app.result_df.columns))
            self.assertEqual(
                export_df.columns[3:].tolist(),
                [
                    col
                    for col in self.app.result_df.columns
                    if col not in {"spectrum_id", "mapped_strain_percent", "mapped_stress_MPa"}
                ],
            )
            self.messages.clear()

        self._configure_strain_mapping(pd.DataFrame({"strain": [0.0, 0.011, 0.022]}))
        self.app.align_strain_max_to_reference.set(True)
        self.app.run_mapping()
        self.root.update()
        export_df = self.app._build_export_dataframe(self.app.result_df)

        self.assertIn("mapped_strain_fraction", export_df.columns)
        self.assertIn("aligned_station_strain_percent", export_df.columns)
        self.assertIn("strain_alignment_factor", export_df.columns)

    def test_strain_alignment_is_disabled_by_default_and_preserves_old_mapping(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.05, 0.10],
                "stress_MPa": [0.0, 500.0, 1000.0],
            }
        )
        self.app.station_df = pd.DataFrame({"strain": [0.0, 0.055, 0.11]})
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRAIN_ONLY)
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.interp_method.set(mapper.METHOD_LINEAR)
        self.app.recommendation_confirmed.set(True)

        self.app.run_mapping()
        self.root.update()

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        self.assertIsNotNone(self.app.result_df)
        self.assertFalse(bool(self.app.result_df["strain_alignment_applied"].iloc[0]))
        self.assertAlmostEqual(self.app.result_df["mapped_strain_fraction"].iloc[-1], 0.11)
        self.assertTrue(math.isnan(self.app.result_df["mapped_stress_MPa"].iloc[-1]))

    def test_strain_alignment_maps_station_strain_before_interpolation(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.0, 0.05, 0.10],
                "stress_MPa": [0.0, 500.0, 1000.0],
            }
        )
        self.app.station_df = pd.DataFrame({"strain": [0.0, 0.055, 0.11]})
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRAIN_ONLY)
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.interp_method.set(mapper.METHOD_LINEAR)
        self.app.align_strain_max_to_reference.set(True)
        self.app.recommendation_confirmed.set(True)

        self.app.run_mapping()
        self.root.update()

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        self.assertIsNotNone(self.app.result_df)
        row = self.app.result_df.iloc[-1]
        self.assertTrue(bool(row["strain_alignment_applied"]))
        self.assertAlmostEqual(row["strain_alignment_factor"], 0.10 / 0.11)
        self.assertAlmostEqual(row["raw_station_strain_fraction"], 0.11)
        self.assertAlmostEqual(row["aligned_station_strain_fraction"], 0.10)
        self.assertAlmostEqual(row["mapped_strain_fraction"], 0.10)
        self.assertAlmostEqual(row["mapped_stress_MPa"], 1000.0)
        self.assertTrue(bool(row["within_reference_range"]))

    def test_start_zeroing_runs_before_strain_alignment_and_interpolation(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.02, 0.07, 0.12],
                "stress_MPa": [10.0, 510.0, 1010.0],
            }
        )
        self.app.station_df = pd.DataFrame({"strain": [0.01, 0.065, 0.12]})
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRAIN_ONLY)
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.interp_method.set(mapper.METHOD_LINEAR)
        self.app.zero_reference.set(True)
        self.app.align_strain_max_to_reference.set(True)
        self.app.recommendation_confirmed.set(True)

        self.app.run_mapping()
        self.root.update()

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        row = self.app.result_df.iloc[-1]
        self.assertTrue(bool(row["start_zero_applied"]))
        self.assertAlmostEqual(row["reference_strain_start_offset_fraction"], 0.02)
        self.assertAlmostEqual(row["reference_stress_start_offset_MPa"], 10.0)
        self.assertAlmostEqual(row["station_strain_start_offset_fraction"], 0.01)
        self.assertAlmostEqual(row["raw_station_strain_fraction"], 0.12)
        self.assertAlmostEqual(row["zeroed_station_strain_fraction"], 0.11)
        self.assertAlmostEqual(row["aligned_station_strain_fraction"], 0.10)
        self.assertAlmostEqual(row["strain_alignment_factor"], 0.10 / 0.11)
        self.assertAlmostEqual(row["mapped_strain_fraction"], 0.10)
        self.assertAlmostEqual(row["mapped_stress_MPa"], 1000.0)
        self.assertAlmostEqual(row["reference_max_strain_percent"], 10.0)
        self.assertAlmostEqual(row["station_max_strain_percent"], 11.0)

    def test_start_zeroing_applies_to_station_stress_before_inverse_mapping(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.02, 0.07, 0.12],
                "stress_MPa": [10.0, 510.0, 1010.0],
            }
        )
        self.app.station_df = pd.DataFrame({"sigma_MPa": [10.0, 510.0, 1010.0]})
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_STRESS_ONLY)
        self.app.station_stress_col.set("sigma_MPa")
        self.app.station_stress_unit.set(mapper.STRESS_MPA)
        self.app.interp_method.set(mapper.METHOD_LINEAR)
        self.app.zero_reference.set(True)
        self.app.recommendation_confirmed.set(True)

        self.app.run_mapping()
        self.root.update()

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        row = self.app.result_df.iloc[-1]
        self.assertTrue(bool(row["start_zero_applied"]))
        self.assertAlmostEqual(row["reference_strain_start_offset_fraction"], 0.02)
        self.assertAlmostEqual(row["reference_stress_start_offset_MPa"], 10.0)
        self.assertAlmostEqual(row["station_stress_start_offset_MPa"], 10.0)
        self.assertAlmostEqual(row["raw_station_stress_MPa"], 1010.0)
        self.assertAlmostEqual(row["zeroed_station_stress_MPa"], 1000.0)
        self.assertAlmostEqual(row["mapped_stress_MPa"], 1000.0)
        self.assertAlmostEqual(row["mapped_strain_fraction"], 0.10)
        self.assertTrue(bool(row["within_reference_range"]))

    def test_start_zeroing_audit_columns_are_exported_in_both_mode(self):
        self.app.ref_df = pd.DataFrame(
            {
                "engineering_strain": [0.02, 0.12],
                "stress_MPa": [10.0, 1010.0],
            }
        )
        self.app.station_df = pd.DataFrame(
            {
                "frame": [1, 2],
                "strain": [0.01, 0.06],
                "sigma_MPa": [20.0, 520.0],
            }
        )
        self.app.ref_strain_col.set("engineering_strain")
        self.app.ref_stress_col.set("stress_MPa")
        self.app.ref_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.ref_stress_unit.set(mapper.STRESS_MPA)
        self.app.station_mode.set(mapper.MODE_BOTH)
        self.app.station_id_col.set("frame")
        self.app.station_strain_col.set("strain")
        self.app.station_strain_unit.set(mapper.STRAIN_FRACTION)
        self.app.station_stress_col.set("sigma_MPa")
        self.app.station_stress_unit.set(mapper.STRESS_MPA)
        self.app.zero_reference.set(True)
        self.app.recommendation_confirmed.set(True)

        self.app.run_mapping()
        self.root.update()
        export_df = self.app._build_export_dataframe(self.app.result_df)

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        row = export_df.iloc[-1]
        for col in [
            "start_zero_applied",
            "reference_strain_start_offset_fraction",
            "reference_stress_start_offset_MPa",
            "station_strain_start_offset_fraction",
            "station_stress_start_offset_MPa",
            "raw_station_strain_fraction",
            "zeroed_station_strain_fraction",
            "aligned_station_strain_fraction",
            "raw_station_stress_MPa",
            "zeroed_station_stress_MPa",
        ]:
            self.assertIn(col, export_df.columns)
        self.assertAlmostEqual(row["raw_station_strain_fraction"], 0.06)
        self.assertAlmostEqual(row["zeroed_station_strain_fraction"], 0.05)
        self.assertAlmostEqual(row["aligned_station_strain_fraction"], 0.05)
        self.assertAlmostEqual(row["raw_station_stress_MPa"], 520.0)
        self.assertAlmostEqual(row["zeroed_station_stress_MPa"], 500.0)
        self.assertAlmostEqual(row["mapped_strain_fraction"], 0.05)
        self.assertAlmostEqual(row["mapped_stress_MPa"], 500.0)

    def test_run_mapping_accepts_station_file_that_already_has_spectrum_id(self):
        self._configure_strain_mapping(
            pd.DataFrame({"spectrum_id": [101, 102, 103], "strain": [0.0, 0.01, 0.02]})
        )

        self.app.run_mapping()
        self.root.update()

        self.assertFalse([msg for msg in self.messages if msg[0] == "error"])
        self.assertIsNotNone(self.app.result_df)
        self.assertEqual(self.app.result_df["spectrum_id"].tolist(), [101, 102, 103])
        self.assertEqual(str(self.app.export_button.cget("state")), "normal")

    def test_failed_rerun_clears_stale_result_and_disables_export(self):
        self._configure_strain_mapping(pd.DataFrame({"strain": [0.0, 0.01, 0.02]}))
        self.app.run_mapping()
        self.root.update()
        self.assertIsNotNone(self.app.result_df)

        self.app.ref_df = pd.DataFrame({"engineering_strain": [0.0], "stress_MPa": [0.0]})
        self.app.run_mapping()
        self.root.update()

        self.assertIsNone(self.app.result_df)
        self.assertEqual(str(self.app.export_button.cget("state")), "disabled")
        self.assertTrue([msg for msg in self.messages if msg[0] == "error"])

    def test_workstation_layout_prioritizes_plot_area_and_moves_logs_to_tab(self):
        self.root.geometry("1280x800")
        self.root.update_idletasks()
        self.root.update()

        self.assertTrue(hasattr(self.app, "left_panel"))
        self.assertTrue(hasattr(self.app, "plot_frame"))
        self.assertTrue(hasattr(self.app, "detail_notebook"))
        self.assertLessEqual(self.app.left_panel.winfo_width(), 440)
        self.assertGreaterEqual(self.app.plot_frame.winfo_width(), 780)
        self.assertIn("结果表", [self.app.detail_notebook.tab(i, "text") for i in self.app.detail_notebook.tabs()])
        self.assertIn("问题日志", [self.app.detail_notebook.tab(i, "text") for i in self.app.detail_notebook.tabs()])

    def test_advanced_settings_are_collapsed_by_default_and_scrollable(self):
        self.root.geometry("1280x800")
        self.root.update_idletasks()
        self.root.update()

        self.assertTrue(hasattr(self.app, "control_canvas"))
        self.assertTrue(hasattr(self.app, "advanced_sections"))
        self.assertFalse(bool(self.app.advanced_visible.get()))
        self.assertFalse(self.app.advanced_frame.winfo_ismapped())
        self.assertGreaterEqual(
            set(self.app.advanced_sections.keys()),
            {"插值与反插值", "平滑显示", "塑性应变对齐", "起点归零 / 基线校正"},
        )
        self.assertTrue(all(not section["body"].winfo_ismapped() for section in self.app.advanced_sections.values()))

        self.app.advanced_visible.set(True)
        self.app._toggle_advanced_settings()
        self.root.update_idletasks()
        self.root.update()

        self.assertTrue(self.app.advanced_frame.winfo_ismapped())
        self.assertTrue(any(section["body"].winfo_ismapped() for section in self.app.advanced_sections.values()))


if __name__ == "__main__":
    unittest.main()
