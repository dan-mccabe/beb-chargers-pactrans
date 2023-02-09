import pandas as pd
import numpy as np
import googlemaps
import requests
import zipfile
import io
import pickle
from collections.abc import Iterable
from datetime import datetime, timedelta


class GTFSData:
    """
    Class for reading and processing GTFS data.
    """
    def __init__(self, dir):
        """
        Constructor for GTFSData class.
        :param dir: directory housing all needed GTFS files.
        """
        block_file = '{}/block.txt'.format(dir)
        block_trip_file = '{}/block_trip.txt'.format(dir)
        trips_file = '{}/trips.txt'.format(dir)
        calendar_file = '{}/calendar.txt'.format(dir)
        calendar_dates_file = '{}/calendar_dates.txt'.format(dir)
        shapes_file = '{}/shapes.txt'.format(dir)
        routes_file = '{}/routes.txt'.format(dir)
        stop_times_file = '{}/stop_times.txt'.format(dir)
        self.block_df = pd.read_csv(block_file)
        self.block_trip_df = pd.read_csv(block_trip_file)
        self.trips_df = pd.read_csv(trips_file)
        self.calendar_df = pd.read_csv(calendar_file)
        self.calendar_dates_df = pd.read_csv(calendar_dates_file)
        self.shapes_df = pd.read_csv(shapes_file)
        self.stop_times_df = pd.read_csv(stop_times_file)
        self.routes_df = pd.read_csv(routes_file, index_col=0)
        self.trip_idx_to_id = dict()

    @staticmethod
    def from_pickle(fname):
        """
        Unpickle a GTFSData object.
        :param fname: filename of pickled GTFSData object
        :return: unpickled object
        """
        with open(fname, 'rb') as f:
            obj = pickle.load(f)
        return obj

    @staticmethod
    def filter_df(df, col, value):
        """
        Filter down the given DataFrame, returning all rows where the
        column "col" has the given value "value"
        :param df: a DataFrame
        :param col: column of DataFrame to filter on
        :param value: desired value in column, may be a single value or
            an iterable
        :return: Filtered DataFrame
        """
        # If value is iterable, check if column value is in it
        if isinstance(value, Iterable):
            return df[df[col].isin(value)]
        # Otherwise, just check equality
        else:
            return df[df[col] == value]

    def pickle(self, fname):
        """
        Pickle this object.
        :param fname: filename to save to
        :return: nothing, just saves file
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def get_trips_in_block(self, block_id):
        """
        Return a DataFrame of trip data for all trips in the given block
        :param block_id: GTFS block ID
        :return:
        """
        # Extract relevant trips
        same_block_trips = self.filter_df(
            self.block_trip_df, 'block_identifier', block_id)
        filtered_trips = self.filter_df(
            self.trips_df, 'trip_id', same_block_trips['trip_id'])

        # Combine trip data from both sources and clean up dataframe
        merged_trips = pd.merge(filtered_trips, same_block_trips, on='trip_id')
        merged_trips = merged_trips.sort_values(by='trip_sequence')
        merged_trips = merged_trips.set_index('trip_id')
        merged_trips = merged_trips.drop(
            columns=['trip_short_name', 'peak_flag', 'fare_id',
                     'service_id_y'])

        # Mark route type of all trips
        merged_trips['route_type'] = merged_trips.apply(
            lambda x: self.routes_df.at[x['route_id'], 'route_type'], axis=1)

        # If any trips are not made by bus, disregard this block
        if any(merged_trips['route_type'].values != 3):
            return None

        else:
            # Add new columns giving route number and trip distance
            merged_trips['route_short_name'] = merged_trips.apply(
                lambda x: self.routes_df.at[x['route_id'], 'route_short_name'],
                axis=1)
            shape_data_df = merged_trips.apply(lambda x: get_trip_data(
                    x.name, x['shape_id'], self.stop_times_df, self.shapes_df),
                axis=1)
            merged_trips = pd.concat([merged_trips, shape_data_df], axis=1)

            return merged_trips

    def build_opt_inputs(self, charging_blocks, block_df_dict,
                         charge_nodes, charge_coords, depot_coords):
        """
        Built inputs for optimization model.
        :return: dictionary of keyword arguments for BEBModel
        """
        trip_start_times = dict()
        trip_start_locs = dict()
        trip_end_times = dict()
        trip_end_locs = dict()
        trip_dists = dict()
        inter_trip_times = dict()
        inter_trip_dists = dict()
        for v in charging_blocks:
            for i, t in enumerate(block_df_dict[v].index):
                # Add 1 to the trip index to account for deadheading
                # from depot, which we model as trip 0
                t_idx = i + 1
                self.trip_idx_to_id[v, t_idx] = t
                fmt_str = '%H:%M:%S'
                t_ref = datetime.strptime('00:00:00', fmt_str)
                try:
                    start_time_str = block_df_dict[v].at[t, 'start_time']
                    start_time = datetime.strptime(start_time_str, fmt_str)
                except ValueError:
                    # We get a ValueError when we're in hour 24, change
                    # it to hour 23 and add 60 minutes.
                    start_time_str = block_df_dict[v].at[t, 'start_time']
                    extra_hrs = int(start_time_str[:2]) - 23
                    start_time_str = '23' + start_time_str[2:]
                    start_time = datetime.strptime(start_time_str, fmt_str) \
                                 + timedelta(hours=extra_hrs)

                trip_start_times[v, t_idx] = \
                    (start_time - t_ref).total_seconds() / 60
                trip_start_locs[v, t_idx] = block_df_dict[v].at[
                    t, 'start_lat_lon']

                try:
                    end_time_str = block_df_dict[v].at[t, 'end_time']
                    end_time = datetime.strptime(end_time_str, fmt_str)
                except ValueError:
                    # We get a ValueError when we're in hour 24. Change
                    # it to hour 23 and add 60 minutes.
                    end_time_str = block_df_dict[v].at[t, 'end_time']
                    extra_hrs = int(end_time_str[:2]) - 23
                    end_time_str = '23' + end_time_str[2:]
                    end_time = datetime.strptime(end_time_str, fmt_str) \
                               + timedelta(hours=extra_hrs)

                trip_end_times[v, t_idx] = \
                    (end_time - t_ref).total_seconds() / 60
                trip_end_locs[v, t_idx] = block_df_dict[v].at[t, 'term_lat_lon']
                trip_dists[v, t_idx] = block_df_dict[v].at[t, 'trip_len']

        # Calculate all necessary distances. Uses two calls, but file is saved
        # in between and reloaded so everything ends up in gmap_charger_data.
        unique_start_locs = list(set(trip_start_locs.values()))
        unique_end_locs = list(set(trip_end_locs.values()))
        charger_locs = list(charge_coords.values()) + [depot_coords]
        _ = get_updated_gmap_data(
            unique_start_locs + unique_end_locs, charger_locs,
            'so_king_cty.pickle')
        gmap_charger_data = get_updated_gmap_data(
            unique_end_locs, unique_start_locs, 'so_king_cty.pickle')

        trip_start_chg_dists = dict()
        trip_start_chg_times = dict()
        trip_end_chg_dists = dict()
        trip_end_chg_times = dict()
        for v in charging_blocks:
            for i, t in enumerate(block_df_dict[v].index):
                # Add 1 to the trip index to account for deadheading
                # from depot, which we model as trip 0
                t_idx = i + 1
                for s in charge_nodes:
                    gmap_start = gmap_charger_data[
                        trip_start_locs[v, t_idx], charge_coords[s]]
                    gmap_end = gmap_charger_data[
                        trip_end_locs[v, t_idx], charge_coords[s]]
                    trip_start_chg_dists[v, t_idx, s] = gmap_start['distance']
                    trip_start_chg_times[v, t_idx, s] = gmap_start['duration']
                    trip_end_chg_dists[v, t_idx, s] = gmap_end['distance']
                    trip_end_chg_times[v, t_idx, s] = gmap_end['duration']

                    if i < len(block_df_dict[v]) - 1:
                        gmap_inter = gmap_charger_data[
                            trip_end_locs[v, t_idx], trip_start_locs[v, t_idx + 1]]
                        inter_trip_dists[v, t_idx] = gmap_inter['distance']
                        inter_trip_times[v, t_idx] = gmap_inter['duration']

            # Add in trips to/from depot at end/start of day.
            gmap_trip_0 = gmap_charger_data[
                trip_start_locs[v, 1], depot_coords]
            trip_dists[v, 0] = gmap_trip_0['distance']
            trip_start_times[v, 0] = trip_start_times[v, 1]
            trip_end_times[v, 0] = trip_start_times[v, 1]
            inter_trip_dists[v, 0] = 0
            inter_trip_times[v, 0] = 0

            end_idx = t_idx + 1
            gmap_last_trip = gmap_charger_data[
                trip_end_locs[v, end_idx-1], depot_coords]
            trip_dists[v, end_idx] = gmap_last_trip['distance']
            # Maintain penalty for late arrival to depot
            trip_start_times[v, end_idx] = trip_end_times[v, end_idx-1]
            trip_end_times[v, end_idx] = trip_end_times[v, end_idx-1]
            inter_trip_dists[v, end_idx-1] = 0
            inter_trip_times[v, end_idx-1] = 0
            inter_trip_dists[v, end_idx] = 0
            inter_trip_times[v, end_idx] = 0

            for s in charge_nodes:
                # Add data for initial trip from depot
                # Don't charge until first real trip is complete (bus
                # fully charged when leaving depot anyway)
                trip_start_chg_dists[v, 0, s] = 1000
                trip_start_chg_times[v, 0, s] = 1000
                trip_end_chg_dists[v, 0, s] = 1000
                trip_end_chg_times[v, 0, s] = 1000

                # Add data for final trip to depot
                # Charging distance/time at the start of depot return
                # trip are the same as at end of final "real" trip
                trip_start_chg_dists[v, end_idx, s] = trip_end_chg_dists[
                    v, end_idx - 1, s]
                trip_start_chg_times[v, end_idx, s] = trip_end_chg_times[
                    v, end_idx - 1, s]
                # No sense in charging after final trip (to depot)
                trip_end_chg_dists[v, end_idx, s] = 1000
                trip_end_chg_times[v, end_idx, s] = 1000

        veh_trip_pairs = list(trip_start_times.keys())
        return {'veh_trip_pairs': veh_trip_pairs,
                'trip_start_times': trip_start_times,
                'trip_end_times': trip_end_times,
                'trip_dists': trip_dists,
                'inter_trip_dists': inter_trip_dists,
                'trip_start_chg_dists': trip_start_chg_dists,
                'trip_end_chg_dists': trip_end_chg_dists,
                'inter_trip_times': inter_trip_times,
                'trip_start_chg_times': trip_start_chg_times,
                'trip_end_chg_times': trip_end_chg_times}


def update_gtfs_data():
    """
    Download updated GTFS data from Metro website. Will place in gtfs/
    directory.
    """
    gtfs_url = 'https://metro.kingcounty.gov/GTFS/google_daily_transit.zip'
    r = requests.get(gtfs_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(path='data/gtfs/')


def get_trip_data(trip_id, shape_id, stop_times_df, shapes_df):
    """
    Get total distance traveled and final stop location of the given
    GTFS shape.

    :param trip_id: GTFS trip ID
    :param shape_id: GTFS shape ID
    :param stop_times_df: DataFrame from GTFS stop_times.txt file
    :param shapes_df: DataFrame from GTFS shapes.txt file
    :return: pandas Series with fields trip_len, end_time, and
        term_lat_lon
    """
    # Extract relevant DataFrame rows
    this_shape_df = shapes_df[shapes_df['shape_id'] == shape_id]
    this_trip_df = stop_times_df[stop_times_df['trip_id'] == trip_id]
    # Determine total distance traveled by last stop
    total_dist = this_trip_df['shape_dist_traveled'].max()
    # Identify first stop and extract its arrival time
    first_stop = this_trip_df[this_trip_df['stop_sequence']
                              == this_trip_df['stop_sequence'].min()]
    start_time = first_stop['arrival_time'].values[0]
    # Identify last stop and extract its departure time
    last_stop = this_trip_df[this_trip_df['stop_sequence']
                             == this_trip_df['stop_sequence'].max()]
    end_time = last_stop['departure_time'].values[0]
    # Extract coordinates of first stop of shape
    start = this_shape_df[this_shape_df['shape_pt_sequence']
                          == this_shape_df['shape_pt_sequence'].min()]
    start_lat = start['shape_pt_lat'].values[0]
    start_lon = start['shape_pt_lon'].values[0]
    # Extract coordinates of final stop of shape
    term = this_shape_df[this_shape_df['shape_pt_sequence']
                         == this_shape_df['shape_pt_sequence'].max()]
    term_lat = term['shape_pt_lat'].values[0]
    term_lon = term['shape_pt_lon'].values[0]
    # Return shape distance in miles (it's given in feet), plus lat/lon
    # of last stop, as a Series
    return pd.Series(
        [total_dist / 5280, start_time, end_time, (start_lat, start_lon),
         (term_lat, term_lon)], index=['trip_len', 'start_time', 'end_time',
                                       'start_lat_lon', 'term_lat_lon'])


def get_route_data(rt):
    """
    Gather data for the specified route from King County Metro's GTFS
    data.

    :param rt: name of route as string, e.g. '44'
    :return: Tuple of (start points of route (for each trip direction),
             trip distances in each direction)
    """
    # Read in necessary data
    # gtfs_folder = 'google_daily_transit-2019-11-18'
    gtfs_folder = 'gtfs'
    cal_df = pd.read_csv('{}/calendar.txt'.format(gtfs_folder))
    rt_df = pd.read_csv('{}/routes.txt'.format(gtfs_folder))
    trip_df = pd.read_csv('{}/trips.txt'.format(gtfs_folder))
    shape_df = pd.read_csv('{}/shapes.txt'.format(gtfs_folder))
    # Pick a typical Monday between 10/21/19 and 3/20/20. Find the
    # corresponding service ID.
    row = cal_df[(cal_df['monday'] == 1) & (cal_df['start_date'] == 20191021)
                 & (cal_df['end_date'] == 20200320)]
    s_id = row['service_id'].values[0]

    # Get the correct route ID
    row = rt_df[rt_df['route_short_name'] == rt]
    rt_id = row['route_id'].values[0]

    # Get all trips that match the above route and service pattern
    all_trips = trip_df[(trip_df['route_id'] == rt_id)
                        & (trip_df['service_id'] == s_id)]
    # We'll use the most commonly run service in each direction.
    dir_0_trips = all_trips[trip_df['direction_id'] == 0]
    dir_0_shape = dir_0_trips.groupby('shape_id').count().idxmax(axis=0)[
        'trip_id']

    dir_1_trips = all_trips[trip_df['direction_id'] == 1]
    dir_1_shape = dir_1_trips.groupby('shape_id').count().idxmax(axis=0)[
        'trip_id']

    # Get the first station for each direction
    shape_0_df = shape_df[shape_df['shape_id'] == dir_0_shape]
    station_0 = shape_0_df[shape_0_df['shape_pt_sequence']
                           == shape_0_df['shape_pt_sequence'].min()]
    station_0_lat_lon = (station_0['shape_pt_lat'].values[0],
                         station_0['shape_pt_lon'].values[0])

    shape_1_df = shape_df[shape_df['shape_id'] == dir_1_shape]
    station_1 = shape_1_df[shape_1_df['shape_pt_sequence']
                           == shape_1_df['shape_pt_sequence'].min()]
    station_1_lat_lon = (station_1['shape_pt_lat'].values[0],
                         station_1['shape_pt_lon'].values[0])

    # Get distance traveled (we'll just look at direction 0)
    # I believe these distances are given in feet
    dist0 = shape_0_df['shape_dist_traveled'].max()
    dist1 = shape_1_df['shape_dist_traveled'].max()

    return [station_0_lat_lon, station_1_lat_lon], [dist0, dist1]


def get_gmap_distance(orig, dest):
    """
    :param orig: Origin coordinates, tuple of (lat, lon) values
    :param dest: Destination coordinates, tuple of (lat, lon) values
    :return: Dict of distance and duration to drive from orig to dest
        and back, as calculated by Google Maps
    """
    # Perform request to use the Google Maps API web service
    # Deliberately putting in an incorrect key so my account doesn't get
    # run up -- if you want to run this function, register for an API
    # key through Google's Developer site. Each account gets a limited
    # amount of free usage each month.
    API_key = 'my_api_key'
    gmaps = googlemaps.Client(key=API_key)

    # pass origin and destination variables to distance_matrix function
    result = gmaps.distance_matrix(orig, dest, mode='driving')

    # Get the distance and time, converting from meters to miles and
    # from seconds to minutes
    dist = result['rows'][0]['elements'][0]['distance']['value'] / 1609
    time = result['rows'][0]['elements'][0]['duration']['value'] / 60

    return {'distance': dist, 'duration': time}


def get_updated_gmap_data(origins, dests,
                          filename='data/gmap_charge_data.pickle'):
    """
    Update time/distance data for charging. Checks for existence of
    data to minimize unnecessary API calls.

    :param origins: Origin coordinates
    :param dests: Destination coordinates
    :param filename: String giving file name to check for existing data
        and write updated data
    :return: Dictionary of all charging data. Also pickles the updated
        dict for future use.
    """
    # Read in saved dict of calculated costs
    try:
        with open(filename, 'rb') as handle:
            charging_travel_data = pickle.load(handle)
    except FileNotFoundError:
        # If file doesn't exist, create new dict
        charging_travel_data = dict()

    for org in origins:
        for dst in dests:
            if (org, dst) not in charging_travel_data:
                charging_travel_data[org, dst] = get_gmap_distance(org, dst)

    with open(filename, 'wb') as handle:
        pickle.dump(charging_travel_data, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    return charging_travel_data


def get_trip_chg_costs(trip_ends, chg_nodes, gmap_data):
    """
    Convert saved travel data with lat/lons to have (trip, node) keys

    :param trip_ends: dict where keys are trip IDs and values are dicts
        with lat/lon pairs of final stop locations 'loc'
    :param chg_nodes: dict where keys are candidate charger location IDs
        and values are lat/lon pairs of locations
    :param gmap_data: dictionary where keys are lat/lon pairs and values
        are dicts of travel distance/time between them, as returned by
        get_updated_gmap_data()
    :return: dict of travel data with (trip, node) keys
    """
    charge_costs = dict()
    for te in trip_ends:
        for chg in chg_nodes:
            charge_costs[chg, te] = gmap_data[trip_ends[te]['loc'],
                                              chg_nodes[chg]]

    return charge_costs


def generate_demands(vehicles, max_charges, trips, trip_ends,
                     distances, e_mu, e_sigma, n_samps):
    """
    Perform a simulation to generate charging demand

    :param vehicles: list of vehicles in operation
    :param max_charges: dict of vehicle battery capacities
    :param trips: dict of t rips for each vehicle
    :param trip_ends: dict giving end locations and times of each trip
    :param distances: dict of distances of each trip
    :param e_mu: mean energy consumption rate
    :param e_sigma: standard deviation of energy consumption rate
    :param n_samps: number of samples
    :return: dictionary giving randomly generated charging demand
    """
    all_trips = [t for k in trips for t in trips[k]]
    hrs = list(set(trip_ends[t]['time'] for t in all_trips))
    end_locs = [trip_ends[t]['loc'] for t in all_trips]
    demand = {(l, h, s): 0 for l in end_locs for h in hrs
              for s in range(n_samps)}
    for s in range(n_samps):
        for v in vehicles:
            e_rates = np.random.normal(loc=e_mu, scale=e_sigma,
                                       size=len(trips[v]))
            charge = max_charges[v]
            for i, t in enumerate(trips[v]):
                # Randomly sample energy consumption
                energy_used = e_rates[i] * distances[t]
                charge -= energy_used
                if charge < 0.1*max_charges[v]:
                    # If battery is under 10%, go charge
                    l = trip_ends[t]['loc']
                    h = trip_ends[t]['time']
                    demand[l, h, s] += 1
                    # Recharge and restart tracking
                    charge = max_charges[v]

    return demand

