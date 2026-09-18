import unittest

from scripts.profile_w3_lut_inline import select_winner


def cell(row_tile, latency, *, stable=True, passed=True):
    return {
        "shape_class": "q_o",
        "row_tile": row_tile,
        "timing_stable": stable,
        "correctness": {
            "passed": passed,
            "repeat_exact": True,
            "workspace_finite": True,
        },
        "timings": {"w3_lut_inline": {"median_us": latency}},
    }


class W3LUTProfileSelectionTest(unittest.TestCase):
    def test_selects_fastest_stable_correct_candidate(self):
        report = {"cells": [cell(256, 3.0), cell(512, 2.0), cell(1024, 4.0)]}
        self.assertEqual(select_winner(report, "q_o")["row_tile"], 512)

    def test_rejects_incomplete_row_tile_coverage(self):
        report = {"cells": [cell(256, 3.0), cell(512, 2.0)]}
        with self.assertRaisesRegex(ValueError, "incomplete row-tile coverage"):
            select_winner(report, "q_o")

    def test_rejects_when_no_candidate_is_eligible(self):
        report = {
            "cells": [
                cell(256, 3.0, stable=False),
                cell(512, 2.0, passed=False),
                {**cell(1024, 4.0), "correctness": {"passed": True, "repeat_exact": False, "workspace_finite": True}},
            ]
        }
        with self.assertRaisesRegex(ValueError, "no stable, correct"):
            select_winner(report, "q_o")


if __name__ == "__main__":
    unittest.main()
