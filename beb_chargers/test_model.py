import unittest
from beb_model import OperationsModel


class TestOperationsModel(unittest.TestCase):
    def setup(self):
        self.op = OperationsModel(
            vehicles=[100, 200], veh_trip_pairs=[(100, 1), (200, 1), (200, 2)],
            chg_sites=['a', 'b'], trip_start_times=None, trip_end_times=None,
            trip_dists=None, inter_trip_dists=None, trip_start_chg_dists=None,
            trip_end_chg_dists=None, inter_trip_times=None, chg_lims=None,
            trip_start_chg_times=None, trip_end_chg_times=None, chg_rates=None,
            energy_rates=None)

    def test_to_csv(self):
        self.op.chg_schedule = {(100, 1, 'a'): 0, (100, 1, 'b'): 0,
                                (200, 1, 'a'): 0, (200, 1, 'b'): 10,
                                (200, 2, 'a'): 0, (200, 2, 'b'): 0}
        expect_veh = [100, 200, 200]
        expect_trip = [1, 1, 2]
        expect_chg_loc = [None, 'b', None]
        expect_chg_time = [0, 10, 0]
        expect_soc = [10, 20, 10]
        expect_delay = [0, 0, 5]
        expect_recov = [2, 4, 2]
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
