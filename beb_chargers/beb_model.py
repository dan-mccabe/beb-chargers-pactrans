import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
import logging
import time
from pyomo.environ import ConcreteModel, Set, Var, Binary, Constraint, \
    Objective, SolverFactory, value, NonNegativeIntegers, NonNegativeReals, \
    SolverStatus, TerminationCondition, ConstraintList


class BEBModel:
    """
    Parent class for all types of BEB models.
    """
    def __init__(self):
        self.trip_start_times = None
        self.trip_end_times = None
        self.veh_trip_pairs = None
        self.chg_sites = None
        self.model = None
        self.solver_status = None
        self.chg_intervals = None
        self.charging_vts = None

    def solve(self, **kwargs):
        raise NotImplementedError

    def process_solver_output(self, results, solved_model):
        if (results.solver.status == SolverStatus.ok) and (
                results.solver.termination_condition
                == TerminationCondition.optimal):
            self.solver_status = 'optimal'
            self.model = solved_model
        # Do something when the solution in optimal and feasible
        elif (results.solver.termination_condition
              == TerminationCondition.infeasible):
            self.solver_status = 'infeasible'
            raise ValueError('Solver found model to be infeasible.')
        # Do something when model is infeasible
        else:
            # Something else is wrong
            self.solver_status = str(results.solver.status)
            self.model = solved_model

    def process_results(self):
        raise NotImplementedError

    def print_results(self):
        raise NotImplementedError

    def plot_chargers(self):
        plt.rcParams.update({'font.size': 16})
        # Get start and end times for plotting
        x_lb = min(self.trip_start_times.values())
        x_ub = max([self.trip_end_times[v, t] + value(self.model.delay[v, t])
                   for (v, t) in self.charging_vts])
        used_sites = [s for s in self.chg_sites
                      if any(value(self.model.chg_binary[v, t, s]) == 1
                             for (v, t) in self.charging_vts)]
        n_sites = len(used_sites)

        fig, ax = plt.subplots(nrows=n_sites, ncols=1, sharex=True,
                               sharey=True, figsize=(9, 8))

        for i, s in enumerate(used_sites):
            start_times = sorted([i[0] for i in self.chg_intervals[s]])
            end_times = sorted([i[1] for i in self.chg_intervals[s]])
            x = [x_lb] + sorted(start_times + end_times) + [x_ub]
            y = [0]*len(x)
            start_idx = 0
            end_idx = 0
            while start_idx < len(start_times) and end_idx < len(end_times):
                y_idx = start_idx + end_idx + 1
                start_val = start_times[start_idx]
                end_val = end_times[end_idx]
                if start_val < end_val:
                    y[y_idx] = y[y_idx-1] + 1
                    start_idx += 1
                elif end_val < start_val:
                    y[y_idx] = y[y_idx-1] - 1
                    end_idx += 1
                else:
                    # New charge begins as soon as another ends
                    y[y_idx] = y[y_idx-1]
                    y[y_idx+1] = y[y_idx]
                    start_idx += 1
                    end_idx += 1

            plt.subplot(n_sites, 1, i+1)
            plt.step(x, y, where='post')
            label = s
            plt.title('Charging Site: {}'.format(label))

        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.ylabel('Number of Vehicles Charging')
        plt.xlabel('Time (min)')
        plt.tight_layout()
        plt.show()


class FacilityLocationModel(BEBModel):
    def __init__(self, vehicles, veh_trip_pairs, chg_sites, chg_lims,
                 trip_start_times, trip_end_times, trip_dists,
                 inter_trip_dists, trip_start_chg_dists, trip_end_chg_dists,
                 inter_trip_times, trip_start_chg_times, trip_end_chg_times,
                 chg_rates, energy_rates, site_costs, q_lambda=1,
                 n_blocks=None):
        super().__init__()
        # Set attributes
        self.vehicles = vehicles
        self.charging_vehs = None
        self.veh_trip_pairs = veh_trip_pairs
        self.charging_vts = None
        self.chg_sites = chg_sites
        self.chg_lims = chg_lims
        self.trip_start_times = trip_start_times
        self.trip_end_times = trip_end_times
        self.trip_dists = trip_dists
        self.inter_trip_dists = inter_trip_dists
        self.trip_start_chg_dists = trip_start_chg_dists
        self.trip_end_chg_dists = trip_end_chg_dists
        self.inter_trip_times = inter_trip_times
        self.trip_start_chg_times = trip_start_chg_times
        self.trip_end_chg_times = trip_end_chg_times
        self.chg_rates = chg_rates
        self.energy_rates = energy_rates
        self.site_costs = site_costs
        self.q_lambda = q_lambda
        self.n_blocks = n_blocks
        self.conflict_vts = list()

        # Initialize result attributes
        self.model = None
        self.solver_status = None
        self.soln_time = None
        self.opt_cost = None
        self.opt_stations = None
        self.opt_charges = None
        self.opt_total_chg_time = None
        self.pct_trips_delayed = None
        self.max_delay = None
        self.opt_delay = None
        self.opt_waiting = None
        self.chg_intervals = None
        self.chgs_per_veh = None
        self.interarrival_times = None
        self.service_times = None
        self.chg_schedule = None
        self.prior_chgs = None

    def check_charging_needs(self):
        # Filter out blocks that don't require charging
        trip_energy_needs = dict()
        for (v, t) in self.veh_trip_pairs:
            try:
                trip_energy_needs[v, t] = self.energy_rates[v, t] * (
                        self.trip_dists[v, t] + self.inter_trip_dists[v, t])
            except KeyError:
                raise KeyError(
                    'Key error for vehicle {}, trip {}'.format(v, t))

        # Identify which vehicles can be excluded because they never
        # need to charge
        trip_energy_needs = {(v, t): self.energy_rates[v, t] * (
                self.trip_dists[v, t] + self.inter_trip_dists[v, t])
                             for (v, t) in self.veh_trip_pairs}
        block_energy_needs = {v: sum(trip_energy_needs[v1, t]
                                     for (v1, t) in self.veh_trip_pairs
                                     if v1 == v) for v in self.vehicles}
        self.charging_vehs = [v for v in self.vehicles
                         if block_energy_needs[v] > self.chg_lims[v]]
        print('Number of blocks that require charging: {}'.format(
            len(self.charging_vehs)))

        # If requested, limit number of vehicles to include in model
        if self.n_blocks is not None:
            self.charging_vehs = self.charging_vehs[:self.n_blocks]
            self.vehicles = self.charging_vehs
            print('Limiting study to first {} blocks'.format
                  (len(self.charging_vehs)))

        self.charging_vts = [(v, t) for (v, t) in self.veh_trip_pairs
                             if v in self.charging_vehs]
        print('Number of trips in charging blocks: {}'.format(
            len(self.charging_vts)))

    def build_cp_model(self, alpha, beta, opt_gap=None):
        """
        Solve BEB charger facility location problem.
        :param alpha: Objective function parameter alpha
        :param beta: Objective function parameter beta
        :param opt_gap: Relative optimality gap for Gurobi MIP solver
        """
        self.check_charging_needs()
        conflict_vts = list()
        for (v, t) in self.charging_vts:
            for (v1, t1) in self.charging_vts:
                if v1 != v:
                    for s in self.chg_sites:
                        conflict_vts.append((v, t, v1, t1, s))

        # Build Pyomo model
        # First, define sets
        m = ConcreteModel()
        m.chg_sites = Set(initialize=self.chg_sites)
        m.vehicles = Set(initialize=self.charging_vehs)
        m.vt_pairs = Set(dimen=2, initialize=self.charging_vts)
        m.conflict_vts = Set(dimen=5, initialize=conflict_vts)

        # Variables
        m.battery_charge = Var(m.vt_pairs, within=NonNegativeReals)
        m.site_binary = Var(m.chg_sites, within=Binary)
        m.chg_binary = Var(m.vt_pairs, m.chg_sites, within=Binary)
        m.chg_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)
        m.delay = Var(m.vt_pairs, within=NonNegativeReals)
        m.wait_time = Var(m.vt_pairs, within=NonNegativeReals)
        m.arr_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)
        m.queue_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)
        m.prec_binary = Var(m.conflict_vts, within=Binary)

        # Big M parameter
        M = 100000

        # Constraints
        m.arrival_prec_1 = ConstraintList()
        m.arrival_prec_2 = ConstraintList()
        m.conflict_1 = ConstraintList()
        m.conflict_2 = ConstraintList()

        def chg_at_built_sites(model, v, t, s):
            return model.chg_binary[v, t, s] <= model.site_binary[s]
        m.x_y_constr = Constraint(m.vt_pairs, m.chg_sites,
                                  rule=chg_at_built_sites)

        def chg_vars_relation(model, v, t, s):
            return model.chg_time[v, t, s] <= M * model.chg_binary[v, t, s]
        m.chg_vars_constr = Constraint(m.vt_pairs, m.chg_sites,
                                       rule=chg_vars_relation)

        def init_charge(model, v, t):
            if t == 0:
                return model.battery_charge[v, t] == self.chg_lims[v]
            else:
                return Constraint.Skip
        m.init_chg_constr = Constraint(m.vt_pairs, rule=init_charge)

        def init_delay(model, v, t):
            if t == 0:
                return model.delay[v, t] == 0
            else:
                return Constraint.Skip
        m.init_delay_constr = Constraint(m.vt_pairs, rule=init_delay)

        def charge_tracking(model, v, t):
            # Don't apply to last trip
            if (v, t + 1) not in self.veh_trip_pairs:
                return Constraint.Skip

            return model.battery_charge[v, t + 1] == model.battery_charge[v, t] \
                   + sum(self.chg_rates[s] * model.chg_time[v, t, s]
                         - self.energy_rates[v, t] * (
                                     self.trip_end_chg_dists[v, t, s]
                                     + self.trip_start_chg_dists[v, t + 1, s])
                         * model.chg_binary[v, t, s] for s in model.chg_sites) \
                   - self.energy_rates[v, t] * self.trip_dists[v, t] \
                   - self.energy_rates[v, t] * self.inter_trip_dists[v, t] * (
                           1 - sum(
                       model.chg_binary[v, t, s] for s in model.chg_sites))
        m.chg_track_constr = Constraint(m.vt_pairs, rule=charge_tracking)

        # Constraints to define queue time
        def arrival_time(model, v, t, s):
            # Set the time a bus arrives at a charger
            return model.arr_time[v, t, s] == model.delay[v, t] \
                   + model.queue_time[v, t, s] + self.trip_end_times[v, t] \
                   + self.trip_end_chg_times[v, t, s] \
                   + M*(1 - model.chg_binary[v, t, s])
        m.arr_time_constr = Constraint(
            m.vt_pairs, m.chg_sites, rule=arrival_time)

        def delay_tracking(model, v, t):
            if (v, t + 1) not in self.veh_trip_pairs:
                return model.wait_time[v, t] == 0

            return model.delay[v, t + 1] == model.delay[v, t] + \
                   self.trip_end_times[v, t] + sum(
                model.chg_time[v, t, s] + model.queue_time[v, t, s] + (
                        self.trip_end_chg_times[v, t, s]
                        + self.trip_start_chg_times[v, t + 1, s])
                * model.chg_binary[v, t, s] for s in self.chg_sites) \
                   + self.inter_trip_times[v, t] * (1 - sum(
                model.chg_binary[v, t, s] for s in self.chg_sites)) \
                   + model.wait_time[v, t] - self.trip_start_times[v, t + 1]
        m.delay_track_constr = Constraint(m.vt_pairs, rule=delay_tracking)

        def charge_min(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] - sum(
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in model.chg_sites)  \
                   >= 0

        m.charge_min_constr = Constraint(m.vt_pairs, rule=charge_min)

        def charge_max(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] + sum(
                self.chg_rates[s] * model.chg_time[v, t, s] -
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in model.chg_sites) - M * (
                           1 - sum(
                       model.chg_binary[v, t, s] for s in model.chg_sites)) \
                   <= self.chg_lims[v]

        m.charge_max_constr = Constraint(m.vt_pairs, rule=charge_max)

        def single_site_charging(model, v, t):
            return sum(model.chg_binary[v, t, s] for s in model.chg_sites) <= 1
        m.single_charge_constr = Constraint(m.vt_pairs,
                                            rule=single_site_charging)

        # Objective function
        def min_cost(model):
            return sum(self.site_costs[s] * model.site_binary[s]
                       for s in model.chg_sites) + alpha * sum(
                model.delay[v, t] - beta * model.wait_time[v, t]
                for (v, t) in model.vt_pairs)
        m.obj = Objective(rule=min_cost)

        solver = SolverFactory('gurobi')
        if opt_gap is not None:
            solver.options['MIPgap'] = opt_gap
        results = solver.solve(m)

        self.process_solver_output(results, m)
        self.process_results()

    def check_charge_conflicts(self):
        new_vt_conflicts = list()
        for s in self.opt_stations:
            charging_vts = [(v, t) for (v, t) in self.charging_vts
                            if value(self.model.chg_binary[v, t, s]) == 1]
            arr_times = {(v, t): value(self.model.arr_time[v, t, s])
                         for (v, t) in charging_vts}
            finish_times = {(v, t): arr_times[v, t] + value(self.model.chg_time[v, t, s])
                            for (v, t) in charging_vts}
            for (v, t) in charging_vts:
                for (v1, t1) in charging_vts:
                    if v1 != v:
                        eps = 1e-6
                        if arr_times[v, t] + eps < arr_times[v1, t1] < finish_times[v, t] - eps:
                            str1 = 'Charging conflict at time {:.2f}:\n'.format(arr_times[v1, t1])
                            str2 = '\tVehicle {} Trip {} charges from {:.2f} to {:.2f}\n'.format(
                                v, t, arr_times[v, t], finish_times[v, t])
                            str3 = '\tVehicle {} Trip {} charges from {:.2f} to {:.2f}\n'.format(
                                v1, t1, arr_times[v1, t1], finish_times[v1, t1])
                            logging.debug(str1 + str2 + str3)
                            new_vt_conflicts.append((v, t, v1, t1, s))

        logging.info('{} charging conflicts detected. '
                     'Adding {} constraints to next iteration'.format(
            len(new_vt_conflicts), len(new_vt_conflicts)))

        return new_vt_conflicts

    def iter_solve(self, alpha, beta, opt_gap=None):
        self.build_cp_model(alpha, beta)

        # Identify charging conflicts
        vt_conflicts = list()
        M = 100000
        eps = 1
        i = 0
        while i < 50:
            i += 1
            logging.info('Solver Iteration {}'.format(i))

            new_vt_conflicts = self.check_charge_conflicts()
            missed_conflicts = [vt for vt in new_vt_conflicts if vt in vt_conflicts]

            # We're done if there are not conflicts
            if not new_vt_conflicts:
                print('No conflicts detected. Solve terminated!')
                break

            elif missed_conflicts:
                for v, t, v1, t1, s in missed_conflicts:
                    logging.error(
                        'Violation found for constraint already added to '
                        'model. Occurred at iteration {}: \n{} '.format(i, (v, t, v1, t1, s)))

                    # Check constraint values for debugging
                    arr_1 = value(self.model.arr_time[v, t, s])
                    arr_2 = value(self.model.arr_time[v1, t1, s])
                    end_1 = arr_1 + value(self.model.chg_time[v, t, s])
                    end_2 = arr_2 + value(self.model.chg_time[v1, t1, s])
                    prec_val = value(self.model.prec_binary[v, t, v1, t1, s])
                    lhs1 = end_1 - arr_2
                    rhs1 = M * (1 - prec_val)
                    lhs2 = end_2 - arr_1
                    rhs2 = M * prec_val
                    str1 = 'Constraint 1: {} - {} + {} <= M(1 - {})'.format(
                        end_1, arr_2, eps, prec_val)
                    eq1 = '\t{} <= {}'.format(lhs1, rhs1)
                    str2 = 'Constraint 2: {} - {} + {} <- M({})'.format(
                        end_2, arr_1, eps, prec_val)
                    eq2 = '\t{} <= {}'.format(lhs2, rhs2)
                    logging.debug(str1)
                    logging.debug(eq1)
                    logging.debug(str2)
                    logging.debug(eq2)
                    raise UserWarning('Constraint added to formulation was found to be violated')

                if len(new_vt_conflicts) == len(missed_conflicts):
                    logging.error(
                        'Terminating solve because only violated constraints'
                        ' were already added to formulation.')
                    break

            vt_conflicts = list(set(vt_conflicts + new_vt_conflicts))
            logging.debug('{} total conflict constraints have now been added '
                         'out of {} possible\n'.format(len(vt_conflicts),
                len(self.chg_sites) * (len(self.charging_vts) ** 2
                                       - len(self.charging_vts))))

            # Add constraints and re-solve
            for (v, t, v2, t2, s) in new_vt_conflicts:
                expr_a1 = self.model.arr_time[v, t, s] - self.model.arr_time[v2, t2, s] <= M * (
                        1 - self.model.prec_binary[v, t, v2, t2, s])
                self.model.arrival_prec_1.add(expr_a1)
                expr_a2 = self.model.arr_time[v2, t2, s] - self.model.arr_time[v, t, s] <= M * (
                    self.model.prec_binary[v, t, v2, t2, s])
                self.model.arrival_prec_2.add(expr_a2)

                expr_c1 = eps + self.model.arr_time[v, t, s] + self.model.chg_time[v, t, s] \
                           - self.model.arr_time[v2, t2, s] <= M * (
                                  1 - self.model.prec_binary[v, t, v2, t2, s])
                self.model.conflict_1.add(expr_c1)
                expr_c2 = eps + self.model.arr_time[v2, t2, s] + self.model.chg_time[v2, t2, s] \
                           - self.model.arr_time[v, t, s] <= M * self.model.prec_binary[
                              v, t, v2, t2, s]
                self.model.conflict_2.add(expr_c2)

            solve_start = time.time()
            solver = SolverFactory('gurobi')
            if opt_gap is not None:
                solver.options['MIPgap'] = opt_gap
            results = solver.solve(self.model)
            self.process_solver_output(results, self.model)
            solve_time = time.time() - solve_start
            logging.debug('Time to solve iteration {}: {:.2f} seconds'.format(
                i, solve_time))
            self.process_results()
            # self.plot_chargers()

    def solve_linear_queue(self, alpha, beta, opt_gap=None):
        """
        Solve BEB charger facility location problem.
        :param alpha: Objective function parameter alpha
        :param beta: Objective function parameter beta
        :param opt_gap: Relative optimality gap for Gurobi MIP solver
        """
        start_time = time.time()
        self.check_charging_needs()

        # Create prior charge sets
        self.prior_chgs = {
            (v, t, s): [(v1, t1) for (v1, t1) in self.charging_vts
                        if self.trip_end_times[v1, t1]
                        + self.trip_end_chg_times[v1, t1, s]
                        <= self.trip_end_times[v, t]
                        + self.trip_end_chg_times[v, t, s]
                        and v1 != v]
            for (v, t) in self.charging_vts for s in self.chg_sites}

        # Build Pyomo model
        # First, define sets
        m = ConcreteModel()
        m.chg_sites = Set(initialize=self.chg_sites)
        m.vehicles = Set(initialize=self.charging_vehs)
        m.vt_pairs = Set(dimen=2, initialize=self.charging_vts)

        # Variables
        m.battery_charge = Var(m.vt_pairs, within=NonNegativeReals)
        m.site_binary = Var(m.chg_sites, within=Binary)
        m.chg_binary = Var(m.vt_pairs, m.chg_sites, within=Binary)
        m.chg_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)
        m.delay = Var(m.vt_pairs, within=NonNegativeReals)
        m.wait_time = Var(m.vt_pairs, within=NonNegativeReals)
        m.arr_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)
        m.min_arr_time = Var(m.chg_sites, within=NonNegativeReals)
        m.min_arr_binary = Var(m.vt_pairs, m.chg_sites, within=Binary)
        m.queue_time = Var(m.vt_pairs, m.chg_sites, within=NonNegativeReals)

        # Big M parameter
        M = 100000

        # Constraints
        def chg_at_built_sites(model, v, t, s):
            return model.chg_binary[v, t, s] <= model.site_binary[s]
        m.x_y_constr = Constraint(m.vt_pairs, m.chg_sites,
                                  rule=chg_at_built_sites)

        def chg_vars_relation(model, v, t, s):
            return model.chg_time[v, t, s] <= M * model.chg_binary[v, t, s]
        m.chg_vars_constr = Constraint(m.vt_pairs, m.chg_sites,
                                       rule=chg_vars_relation)

        def init_charge(model, v, t):
            if t == 0:
                return model.battery_charge[v, t] == self.chg_lims[v]
            else:
                return Constraint.Skip
        m.init_chg_constr = Constraint(m.vt_pairs, rule=init_charge)

        def init_delay(model, v, t):
            if t == 0:
                return model.delay[v, t] == 0
            else:
                return Constraint.Skip
        m.init_delay_constr = Constraint(m.vt_pairs, rule=init_delay)

        def charge_tracking(model, v, t):
            # Don't apply to last trip
            if (v, t + 1) not in self.veh_trip_pairs:
                return Constraint.Skip

            return model.battery_charge[v, t + 1] == model.battery_charge[v, t] \
                   + sum(self.chg_rates[s] * model.chg_time[v, t, s]
                         - self.energy_rates[v, t] * (
                                     self.trip_end_chg_dists[v, t, s]
                                     + self.trip_start_chg_dists[v, t + 1, s])
                         * model.chg_binary[v, t, s] for s in model.chg_sites) \
                   - self.energy_rates[v, t] * self.trip_dists[v, t] \
                   - self.energy_rates[v, t] * self.inter_trip_dists[v, t] * (
                           1 - sum(
                       model.chg_binary[v, t, s] for s in model.chg_sites))
        m.chg_track_constr = Constraint(m.vt_pairs, rule=charge_tracking)

        # Constraints to define queue time
        def arrival_time(model, v, t, s):
            # Set the time a bus arrives at a charger
            return model.arr_time[v, t, s] == model.delay[v, t] \
                   + self.trip_end_times[v, t] \
                   + self.trip_end_chg_times[v, t, s] \
                   + M*(1 - model.chg_binary[v, t, s])
        m.arr_time_constr = Constraint(
            m.vt_pairs, m.chg_sites, rule=arrival_time)

        def min_arrival_time_1(model, v, t, s):
            # Set the earliest bus arrival at a given charger
            return model.min_arr_time[s] <= model.arr_time[v, t, s]
        m.min_arr_constr_1 = Constraint(
            m.vt_pairs, m.chg_sites, rule=min_arrival_time_1)

        def min_arrival_time_2(model, v, t, s):
            # Set the earliest bus arrival at a given charger
            return model.min_arr_time[s] >= model.arr_time[v, t, s] - 2 * M * (
                1 - model.min_arr_binary[v, t, s])
        m.min_arr_constr_2 = Constraint(
            m.vt_pairs, m.chg_sites, rule=min_arrival_time_2)

        def min_arr_binary_sum(model, s):
            # Ensure only one v-t pair is marked as the earliest arrival
            return sum(model.min_arr_binary[v, t, s] for (v, t) in
                       model.vt_pairs) == 1
        m.min_arr_sum_constr = Constraint(m.chg_sites, rule=min_arr_binary_sum)

        def queue_time(model, v, t, s):
            return model.queue_time[v, t, s] >= self.q_lambda * sum(
                model.chg_time[v1, t1, s] for (v1, t1) in self.prior_chgs[v, t, s]
            ) - (self.trip_end_times[v, t] + self.trip_end_chg_times[v, t, s])\
                   - M*(1 - model.chg_binary[v, t, s]) + model.min_arr_time[s]
            # return model.queue_time[v, t, s] >= 0
        m.q_time_constr = Constraint(m.vt_pairs, m.chg_sites, rule=queue_time)

        def delay_tracking(model, v, t):
            if (v, t + 1) not in self.veh_trip_pairs:
                return model.wait_time[v, t] == 0

            return model.delay[v, t + 1] == model.delay[v, t] + \
                   self.trip_end_times[v, t] + sum(
                model.chg_time[v, t, s] + model.queue_time[v, t, s] + (
                        self.trip_end_chg_times[v, t, s]
                        + self.trip_start_chg_times[v, t + 1, s])
                * model.chg_binary[v, t, s] for s in self.chg_sites) \
                   + self.inter_trip_times[v, t] * (1 - sum(
                model.chg_binary[v, t, s] for s in self.chg_sites)) \
                   + model.wait_time[v, t] - self.trip_start_times[v, t + 1]
        m.delay_track_constr = Constraint(m.vt_pairs, rule=delay_tracking)

        def charge_min(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] - sum(
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in model.chg_sites)  \
                   >= 0

        m.charge_min_constr = Constraint(m.vt_pairs, rule=charge_min)

        def charge_max(model, v, t):
            return model.battery_charge[v, t] - self.energy_rates[v, t] * \
                   self.trip_dists[v, t] + sum(
                self.chg_rates[s] * model.chg_time[v, t, s] -
                self.energy_rates[v, t] * self.trip_end_chg_dists[v, t, s] *
                model.chg_binary[v, t, s] for s in model.chg_sites) - M * (
                           1 - sum(
                       model.chg_binary[v, t, s] for s in model.chg_sites)) \
                   <= self.chg_lims[v]

        m.charge_max_constr = Constraint(m.vt_pairs, rule=charge_max)

        def single_site_charging(model, v, t):
            return sum(model.chg_binary[v, t, s] for s in model.chg_sites) <= 1
        m.single_charge_constr = Constraint(m.vt_pairs,
                                            rule=single_site_charging)

        # Objective function
        def min_cost(model):
            return sum(self.site_costs[s] * model.site_binary[s]
                       for s in model.chg_sites) + alpha * sum(
                model.delay[v, t] - beta * model.wait_time[v, t]
                for (v, t) in model.vt_pairs)
        m.obj = Objective(rule=min_cost)

        solver = SolverFactory('gurobi')
        if opt_gap is not None:
            solver.options['MIPgap'] = opt_gap
        solver.options['TimeLimit'] = 1800
        results = solver.solve(m)
        self.soln_time = time.time() - start_time

        # Check solver status
        self.process_solver_output(results, m)

    def process_results(self):
        # Get objective function value
        self.opt_cost = value(self.model.obj)

        # Which stations are built?
        self.opt_stations = [s for s in self.chg_sites
                             if value(self.model.site_binary[s]) == 1]

        # When and where does charging happen?
        self.chg_schedule = {(v, t, s): value(self.model.chg_time[v, t, s])
                             for (v, t) in self.charging_vts
                             for s in self.chg_sites}

        # Calculate total time spent charging
        self.opt_total_chg_time = sum(value(self.model.chg_time[v, t, s])
                                for (v, t) in self.charging_vts
                                for s in self.chg_sites)

        # Calculate delay metrics
        all_delays = [value(self.model.delay[v, t])
                      for (v, t) in self.charging_vts]
        self.pct_trips_delayed = len([d for d in all_delays if d > 0]) / len(
            all_delays) * 100
        self.max_delay = max(all_delays)
        self.opt_delay = sum(all_delays)

        # Calculate total waiting time
        self.opt_waiting = sum(value(self.model.wait_time[v, t])
                               for (v, t) in self.charging_vts)

        # Calculate each bus's charge over time
        self.opt_charges = sum(value(self.model.chg_binary[v, t, s])
                               for (v, t) in self.charging_vts
                               for s in self.chg_sites)

        # Calculate charger usage
        chg_intervals = {s: list() for s in self.chg_sites}
        chg_idxs = [(v, t, s) for (v, t) in self.charging_vts
                    for s in self.chg_sites
                    if value(self.model.chg_binary[v, t, s]) == 1]
        for (v, t, s) in chg_idxs:
            station_arr = self.trip_end_times[v, t] + self.trip_end_chg_times[
                v, t, s] + value(self.model.delay[v, t]) + value(
                self.model.queue_time[v, t, s])
            station_dept = station_arr + value(self.model.chg_time[v, t, s])
            chg_intervals[s].append((station_arr, station_dept))
        self.chg_intervals = chg_intervals

        # Calculate interarrival times
        sorted_chg_starts = {s: sorted(i[0] for i in chg_intervals[s])
                             for s in self.chg_sites}
        self.interarrival_times = {s: np.diff(sorted_chg_starts[s]).tolist()
                                   for s in self.opt_stations}

        # How long are the charges at each station?
        self.service_times = {s: [value(self.model.chg_time[v, t, s])
                                  for (v, t) in self.charging_vts
                                  if value(self.model.chg_binary[v, t, s]) > 0]
                              for s in self.opt_stations}

        self.chgs_per_veh = {v: sum(value(
            self.model.chg_binary[v, t, s])
            for (v2, t) in self.charging_vts if v2 == v
            for s in self.chg_sites) for v in self.vehicles}

    def solve(self, alpha, beta, opt_gap=None):
        start_time = time.time()
        # self.solve_single_iter(alpha, beta, opt_gap)
        # self.process_results()
        self.iter_solve(alpha, beta)
        self.soln_time = time.time() - start_time
        logging.info('Time for full model solve: {:.2f} seconds'.format(
            self.soln_time))

    def print_results(self):
        print('Optimal objective function value: {:.2f}'.format(self.opt_cost))
        print('Optimal stations: {}'.format(self.opt_stations))
        print('Optimal number of charger visits: {}'.format(self.opt_charges))
        print('Optimal total charging time: {:.2f} minutes'.format(
            self.opt_total_chg_time))
        print('Average time per charge: {:.2f} minutes'.format(
            self.opt_total_chg_time / self.opt_charges))
        print('Optimal total delay: {:.2f} minutes'.format(self.opt_delay))
        print('Optimal maximum delay: {:.2f} minutes'.format(self.max_delay))
        print('Average delay per trip: {:.2f} minutes'.format(
            self.opt_delay / len(self.veh_trip_pairs)))
        print('Percentage of trips delayed: {:.2f}%'.format(
            self.pct_trips_delayed))
        print('Optimal total waiting time: {:.2f} minutes'.format(
            self.opt_waiting))

    def plot_queue_fit(self):
        for s in self.opt_stations:
            # Fit exponential model for interarrival times
            arr_params = stats.expon.fit(self.interarrival_times[s], floc=0)
            print(s, arr_params)
            print(np.mean(self.interarrival_times[s]))
            fig, ax = plt.subplots()
            ax.hist(self.interarrival_times[s], normed=True)
            xlims = ax.get_xlim()
            xvals = np.linspace(*xlims, 100)
            yvals = stats.expon.pdf(xvals, *arr_params)
            ax.plot(xvals, yvals)
            plt.title('Interarrival Times for {}'.format(s))

            # Fit exponential model for service times
            serv_params = stats.expon.fit(self.service_times[s], floc=0)
            print(s, serv_params)
            print(np.mean(self.service_times[s]))
            fig, ax = plt.subplots()
            ax.hist(self.service_times[s], normed=True)
            xlims = ax.get_xlim()
            xvals = np.linspace(*xlims, 100)
            yvals = stats.expon.pdf(xvals, *serv_params)
            ax.plot(xvals, yvals)
            plt.title('Service Times for {}'.format(s))

            plt.show()

    def pickle(self, fname: str):
        """
        Save the model object with pickle. This solved model can then be
        analyzed further with the simulation code without having to
        solve it again from scratch.

        :param fname: filename to save to
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def to_csv(self, fname, trip_id_dict):
        if self.solver_status != 'optimal':
            raise ValueError('Cannot output solution when solver status was '
                             'not optimal.')

        v_list = list()
        t_list = list()
        t_ids = list()
        opt_sites = list()
        opt_arr = list()
        opt_chgs = list()
        opt_soc = list()
        opt_delay = list()
        opt_recov = list()
        for i, (v, t) in enumerate(self.charging_vts):
            v_list.append(v)
            t_list.append(t)
            try:
                t_ids.append(trip_id_dict[v, t])
            except KeyError:
                # KeyError happens when t == 0 (departure from depot)
                # or t == max for this block (return to depot)
                if t == 0:
                    # Use ID 0 as placeholder for departure from depot
                    t_ids.append(0)
                else:
                    # Use ID 100 as placeholder for return to depot
                    t_ids.append(100)

            chg_by_site = {
                s: self.chg_schedule[v, t, s] for s in self.chg_sites}
            if max(chg_by_site.values()) > 1e-6:
                site_i = max(chg_by_site, key=chg_by_site.get)
                chg_i = self.chg_schedule[v, t, site_i]
                arr_i = value(self.model.arr_time[v, t, site_i])
            else:
                site_i = np.nan
                chg_i = np.nan
                arr_i = np.nan

            opt_sites.append(site_i)
            opt_chgs.append(chg_i)
            opt_arr.append(arr_i)
            opt_soc.append(value(self.model.battery_charge[v, t]))
            opt_delay.append(value(self.model.delay[v, t]))
            opt_recov.append(value(self.model.wait_time[v, t]))

        out_dict = {
            'block_id': v_list,
            'trip_id': t_ids,
            'trip_idx': t_list,
            'soc': opt_soc,
            'delay': opt_delay,
            'recovery': opt_recov,
            'chg_site': opt_sites,
            'arrival_time': opt_arr,
            'chg_time': opt_chgs}
        out_df = pd.DataFrame(out_dict)
        out_df.to_csv(fname, index=False)

    def summary_to_csv(self, fname, param_name, param_val):
        if self.solver_status != 'optimal':
            raise ValueError('Cannot output solution when solver status was '
                             'not optimal.')
        out_dict = {
            'obj_val': self.opt_cost,
            'delay_total': self.opt_delay,
            'delay_max': self.max_delay,
            'num_stations': len(self.opt_stations),
            'num_charges': self.opt_charges,
            'charge_time': self.opt_total_chg_time,
            'wait_time': self.opt_waiting,
            param_name: param_val}
        out_df = pd.DataFrame(out_dict, index=[0])
        out_df['stations'] = None
        out_df.at[0, 'stations'] = self.opt_stations
        out_df.to_csv(fname, index=False)



