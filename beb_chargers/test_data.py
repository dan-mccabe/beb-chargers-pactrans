from unittest import TestCase
from unittest.mock import patch, MagicMock
from beb_data import get_updated_gmap_data, get_trip_chg_costs
import pickle
import os


class TestGmapDataUpdate(TestCase):
    def setUp(self):
        self.test_fname = 'test_data_fake.pickle'
        self.test_gmap_fname = 'test_data_gmap.pickle'
        self.test_dict = {
            ((45, 45), (45, 47)): {'distance': 1000, 'duration': 1000},
            ((45, 45), (46, 47)): {'distance': 2000, 'duration': 2000}}

        # Write test dict to pickle file
        with open(self.test_fname, 'wb') as handle:
            pickle.dump(
                self.test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def test_gmaps_called(self):
        # trip_ends = [(47.661351, -122.342091), (47.662133, -122.353972)]
        # chg_locs = [(47.661394, -122.333712)]
        #
        # with patch('googlemaps.Client') as mocked_client:
        #     mocked_client.__reduce__ = lambda s: MagicMock, ()
        #     get_updated_gmap_data(
        #         trip_ends=trip_ends, charge_nodes=chg_locs,
        #         filename=self.test_gmap_fname)
        #     mocked_client.distance_matrix.assert_called()
        return self.fail()

    def test_no_saved_data(self):
        """
        If there is no pre-pickled dict, we should freshly calculate all
        values and save and return the result.
        """
        test_ods = [((47.661351, -122.342091), (47.661394, -122.333712)),
                    ((47.662133, -122.353972), (47.661394, -122.333712))]
        trip_ends = [(47.661351, -122.342091), (47.662133, -122.353972)]
        chg_locs = [(47.661394, -122.333712)]

        travel_data = get_updated_gmap_data(
            origins=trip_ends, dests=chg_locs, filename=self.test_gmap_fname)

        # Confirm we have results for all OD pairs
        for od in test_ods:
            self.assertIn(od, travel_data.keys())
            self.assertIsInstance(travel_data[od]['distance'], float)
            self.assertIsInstance(travel_data[od]['duration'], float)

        # TODO: learn how to mock this properly
        # googlemaps.Client.distance_matrix = MagicMock()
        # assert googlemaps.Client.distance_matrix.called

        # # Confirm we called distance_matrix to get the result
        # with patch('googlemaps.Client.distance_matrix') as myvar:
        #     myvar.assert_called()

        # Confirm results saved correctly
        with open(self.test_gmap_fname, 'rb') as handle:
            saved_data = pickle.load(handle)

        self.assertEqual(travel_data, saved_data)

    def test_all_data_saved(self):
        trip_ends = [(45, 45)]
        chg_locs = [(45, 47), (46, 47)]

        travel_data = get_updated_gmap_data(
            trip_ends=trip_ends, charge_nodes=chg_locs,
            filename=self.test_fname
        )

        self.assertEqual(self.test_dict, travel_data)

        # TODO: learn how to mock this properly
        # Confirm we don't call Google API (and get charged money)
        # with patch('googlemaps.Client.distance_matrix') as myvar:
        #     myvar.assert_not_called()

    def tearDown(self):
        os.remove(self.test_fname)
        try:
            os.remove(self.test_gmap_fname)
        except FileNotFoundError:
            pass


class TestGetTripChgCosts(TestCase):
    def test_simple(self):
        gmap_data = {
            ((45, 45), (45, 47)): {'distance': 1000, 'duration': 1000},
            ((45, 45), (46, 47)): {'distance': 2000, 'duration': 2000}}
        trip_ends = {1: (45, 45)}
        chg_nodes = {'a': (45, 47), 'b': (46, 47)}
        expect = {(1, 'a'): {'distance': 1000, 'duration': 1000},
                  (1, 'b'): {'distance': 2000, 'duration': 2000}}

        result = get_trip_chg_costs(trip_ends, chg_nodes, gmap_data)

