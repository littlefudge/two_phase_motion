import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_main", ROOT / "main_test_rk45.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FixedTests(unittest.TestCase):
    def test_mpi_share_covers_all_points_exactly_once(self):
        for size in (1, 7, 50):
            intervals = [MODULE.get_share(size, rank, 8, 9, 3)
                         for rank in range(size)]
            covered = []
            for start, end in intervals:
                covered.extend(range(start, end))
            self.assertEqual(sorted(covered), list(range(8 * 9 * 3)))

    def test_flat_grid_index_round_trip(self):
        nx, ny, nz = 8, 9, 3
        for index in range(nx * ny * nz):
            i, j, k = MODULE.get_pos(nx, ny, nz, index)
            flat = i * ny * nz + j * nz + k
            self.assertEqual(flat, index)

    def test_hydro_initial_rhs_is_finite(self):
        result = MODULE.hydro(0.0, [1.0, 0.5, 0.1], 1.0, 4.0, 1.0)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(math.isfinite(value) for value in result))

    def test_log_and_linear_grid_building(self):
        params = {
            "n_u": 3, "n_v": 2, "n_r": 2,
            "L_u0": -1, "L_u1": 1,
            "L_v0": -1, "L_v1": 0,
            "L_r0": -2, "L_r1": 0,
            "logu": 1, "logv": 1, "logr": 0,
            "const": {"a": 1.0, "mu": 4.0, "tau": 1.0},
        }
        u, v, r = MODULE.build_grid(params)
        self.assertEqual(len(u), 3)
        self.assertEqual(len(v), 2)
        self.assertEqual(r, [-2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
