from beb_data import GTFSData
from evaluation import SimulationRun
from beb_model import FacilityLocationModel
from pyomo.environ import value
import matplotlib.pyplot as plt
import pandas as pd
import time


def batch_run_model(
        route_list, site_file, battery_cap, kwh_per_mile, charge_power,
        fixed_cost, alpha, beta, lam_vals, save_prefix=None, opt_gap=None):
    # Read in all the data
    gtfs = GTFSData(dir='data/gtfs')

    # Load candidate charging sites given by Metro
    loc_df = pd.read_csv(site_file)
    charge_nodes = list(loc_df['Name'])
    charge_nodes = [charge_nodes[2]]
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

    bus_caps = {b: battery_cap for b in charging_blocks}
    charger_rates = {s: charge_power for s in charge_nodes}
    fixed_costs = {s: fixed_cost for s in charge_nodes}
    energy_rates = {k: kwh_per_mile for k in opt_kwargs['veh_trip_pairs']}

    predicted_delay_vals = list()
    actual_delay_vals = list()
    n_stations_vals = list()
    obj_vals = list()
    for i, lam in enumerate(lam_vals):
        print('\nLambda:', lam)
        flm = FacilityLocationModel(
            vehicles=charging_blocks, chg_sites=charge_nodes, chg_lims=bus_caps,
            chg_rates=charger_rates, energy_rates=energy_rates,
            site_costs=fixed_costs, q_lambda=lam, **opt_kwargs)

        solve_start = time.time()
        flm.solve_linear_queue(alpha=alpha, beta=beta, opt_gap=opt_gap)
        flm.process_results()
        solve_time = time.time() - solve_start
        print('Time to solve: {:.2f} seconds'.format(solve_time))
        # flm.print_results()

        queue_times = {(v, t, s): value(flm.model.queue_time[v, t, s])
                       for (v, t) in flm.charging_vts for s in flm.chg_sites
                       if value(flm.model.queue_time[v, t, s]) > 1e-6}
        predicted_delay_vals.append(sum(queue_times.values()))
        # flm.plot_chargers()
        if save_prefix is not None:
            save_fname = '{}_lambda{}.csv'.format(save_prefix, lam)
            flm.to_csv(save_fname, gtfs.trip_idx_to_id)

        sched = flm.chg_schedule
        # chg_sites = op.opt_stations
        # op.veh_trip_pairs = op.charging_vts
        site_cap = {s: 1 for s in flm.chg_sites}
        sim = SimulationRun(om=flm, chg_plan=sched, site_caps=site_cap)
        # print('\nSIMULATION RESULTS')
        sim.run_sim()
        sim.process_results()
        # sim.print_results()
        actual_delay_vals.append(sim.total_queue_delay)
        n_stations_vals.append(len(flm.opt_stations))
        obj_vals.append(
            sum(fixed_costs[s] for s in flm.opt_stations) + alpha*(
                sim.total_delay - beta*sim.total_recovery))

        # # Update plots every 3 runs
        # if (i+1) % 3 == 0 or i == len(lam_vals) - 1:

    queue_time_errors = [predicted_delay_vals[j] - actual_delay_vals[j]
                         for j in range(len(actual_delay_vals))]
    plt.plot(lam_vals[:i+1], queue_time_errors)
    plt.title('Impact of $\lambda$')
    plt.xlabel('Queue weighting parameter $\lambda$')
    plt.ylabel('Total queue delay error (minutes)')

    plt.figure()
    plt.plot(lam_vals[:i+1], n_stations_vals)
    plt.title('Impact of $\lambda$')
    plt.xlabel('Queue weighting parameter $\lambda$')
    plt.ylabel('Optimal number of stations built')

    plt.figure()
    plt.plot(lam_vals[:i+1], obj_vals)
    plt.title('Impact of $\lambda$')
    plt.xlabel('Queue weighting parameter $\lambda$')
    plt.ylabel('Realized objective function value')
    plt.show()


# Define inputs
# CSV file giving candidate charger sites
site_fname = 'data/so_king_cty_sites.csv'
# Routes to be included
route_list = [101, 102, 111, 116, 143, 150, 157, 158, 159, 177,
              178, 179, 180, 190, 192, 193, 197, 952]
# route_list = [101, 102]
# Battery capacity in kWh
battery_cap = 466 * 0.9
# Energy consumption rate in kWh per mile
kwh_per_mile = 3
# Power output of each charger
chg_pwr = 450/60
# Charger construction cost
fixed_cost = 20
alpha = 1
beta = 0.1
save_name = 'results/queue_lambda/all_rts_'

# lam_vals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
#             1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6,
#             2.7, 2.8, 2.9, 3.0]
lam_vals = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
# lam_vals = [3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5, 5.2, 5.4, 5.6,
#             5.8, 6]
# lam_vals = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]
batch_run_model(
    route_list=route_list, site_file=site_fname, battery_cap=battery_cap,
    kwh_per_mile=kwh_per_mile, charge_power=chg_pwr, fixed_cost=fixed_cost,
    alpha=alpha, beta=beta, lam_vals=lam_vals, opt_gap=0.1, save_prefix=None)
