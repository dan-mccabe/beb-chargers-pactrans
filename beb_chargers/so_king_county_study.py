import pandas as pd
from beb_model import FacilityLocationModel
from beb_data import GTFSData
from pyomo.environ import value
import time
import pickle
import logging


def run_facility_location(
        route_list, site_file, battery_cap, kwh_per_mile, charge_power,
        fixed_cost, alpha, beta, q_lambda=1, pickle_fname=None, save_fname=None,
        opt_gap=None, n_blocks=None, method='lq'):
    # Read in all the data
    gtfs = GTFSData(dir='data/gtfs')

    # Load candidate charging sites given by Metro
    loc_df = pd.read_csv(site_file)
    charge_nodes = list(loc_df['Name'])
    charge_coords = dict()
    for n in charge_nodes:
        subdf = loc_df[loc_df['Name'] == n]
        charge_coords[n] = (float(subdf['y'].values[0]),
                            float(subdf['x'].values[0]))

    # Select routes that will be served by 60' BEBs
    route_list = [str(r) for r in route_list]
    matching_routes = gtfs.filter_df(gtfs.routes_df, 'route_short_name',
                                     route_list)
    route_ids = set(matching_routes.index)

    # Consider all unique blocks that involve these routes
    route_trips = gtfs.filter_df(gtfs.trips_df, 'route_id', route_ids)
    all_blocks = route_trips['block_id'].unique().tolist()
    print('Number of blocks identified:', len(all_blocks))
    route_trips = gtfs.filter_df(route_trips, 'service_id', 86021)
    all_blocks = route_trips['block_id'].unique().tolist()
    print('After limiting service ID:', len(all_blocks))

    block_df_dict = {b: gtfs.get_trips_in_block(b) for b in all_blocks}
    charging_blocks = all_blocks

    all_trips_df = pd.concat([block_df_dict[b] for b in charging_blocks])
    print('Number of trips identified:', len(all_trips_df))

    # Build inputs for optimization
    depot_coords = (47.495809, -122.286190)
    opt_kwargs = gtfs.build_opt_inputs(
        charging_blocks, block_df_dict, charge_nodes,
        charge_coords, depot_coords)

    new_nodes = list()
    fixed_costs = {s: fixed_cost for s in charge_nodes}

    if pickle_fname is not None:
        with open(save_fname, 'wb') as f:
            pickle.dump(opt_kwargs, f)

    bus_caps = {b: battery_cap for b in charging_blocks}
    charger_rates = {s: charge_power for s in charge_nodes}
    energy_rates = {k: kwh_per_mile for k in opt_kwargs['veh_trip_pairs']}

    flm = FacilityLocationModel(
        vehicles=charging_blocks, chg_sites=charge_nodes, chg_lims=bus_caps,
        chg_rates=charger_rates, energy_rates=energy_rates,
        site_costs=fixed_costs, q_lambda=q_lambda, n_blocks=n_blocks,
        **opt_kwargs)

    solve_start = time.time()
    if method == 'lq':
        flm.solve_linear_queue(alpha=alpha, beta=beta, opt_gap=opt_gap)
    else:
        flm.solve(alpha=alpha, beta=beta, opt_gap=opt_gap)
    solve_time = time.time() - solve_start
    flm.process_results()
    print('Time to solve: {:.2f} seconds'.format(solve_time))
    flm.print_results()
    flm.plot_chargers()
    if save_fname is not None:
        flm.to_csv(save_fname, gtfs.trip_idx_to_id)
    return flm


if __name__ == '__main__':
    from evaluation import SimulationRun, SimulationBatch
    logging.basicConfig(level=logging.INFO)

    # Define inputs
    # CSV file giving candidate charger sites
    site_fname = 'data/so_king_cty_sites.csv'
    # Routes to be included
    route_list = [101, 102, 111, 116, 143, 150, 157, 158, 159, 177,
                  178, 179, 180, 190, 192, 193, 197, 952]
    # Battery capacity in kWh
    battery_cap = 466 * 0.9
    # Energy consumption rate in kWh per mile
    kwh_per_mile = 3
    # Power output of each charger
    chg_pwr = 300/60
    # Charger construction cost
    fixed_cost = 20
    alpha = 1
    beta = 0.1

    flm = run_facility_location(
        route_list=route_list, site_file=site_fname, battery_cap=battery_cap,
        kwh_per_mile=kwh_per_mile, charge_power=chg_pwr, fixed_cost=fixed_cost,
        alpha=alpha, beta=beta, q_lambda=1.5, method='lq')
    print('Total queue time: {}'.format(
        sum(value(flm.model.queue_time[v, t, s])
            for (v, t) in flm.charging_vts for s in flm.chg_sites)
    ))

    # Extract needed outputs
    sched = flm.chg_schedule
    # chg_sites = op.opt_stations
    # op.veh_trip_pairs = op.charging_vts
    site_cap = {s: 1 for s in flm.chg_sites}
    sim = SimulationRun(om=flm, chg_plan=sched, site_caps=site_cap)
    print('\nEVALUATION RESULTS')
    sim.run_sim()
    sim.process_results()
    sim.print_results()

    # Batch simulation
    batch = SimulationBatch(om=flm, sched=sched, n_sims=100,
                            site_caps=site_cap, energy_std=0.01,
                            seed=17)
    batch.run()
    print('\nSIMULATION RESULTS')
    batch.process_results()


