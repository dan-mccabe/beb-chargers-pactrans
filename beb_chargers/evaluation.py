import operator
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as tstat


class Vehicle:
    def __init__(self, id, max_chg):
        self.id = id
        self.max_chg = max_chg
        self.chg = max_chg


class ChargingStation:
    def __init__(self, id, power, num_chargers):
        self.id = id
        self.power = power
        self.num_chargers = num_chargers
        # List of ChargingRequests currently charging
        self.reqs_charging = list()
        # List of ChargeRequests in queue
        self.queue = list()

    def is_full(self):
        return len(self.reqs_charging) >= self.num_chargers

    def start_charging(self, req):
        if req in self.reqs_charging:
            raise ValueError('Given vehicle is already charging.')
        self.reqs_charging.append(req)

    def add_to_queue(self, req):
        """
        Add a ChargeRequest to the queue.
        :param req: ChargeRequest instance
        """
        self.queue.append(req)

    def finish_charging(self, v_id):
        """
        Remove a vehicle from the currently charging list
        :param v_id: Vehicle ID
        :return:
        """
        req_to_finish = get_object_by_id(v_id, self.reqs_charging)
        self.reqs_charging.remove(req_to_finish)

    def advance_queue(self):
        """
        Progress through queue if one exists.
        :return: request made by vehicle now charging
        """
        if len(self.queue) > 0:
            now_charging_req = self.queue.pop(0)
            self.start_charging(now_charging_req)
            return now_charging_req

        else:
            return None

    def get_queue_time(self):
        """
        Return how long a vehicle would have to wait to start charging
        if it entered the queue at the current time.
        :return:
        """
        chg_times_left = [r.chg_time for r in self.reqs_charging]
        pseudo_q = copy.copy(self.queue)
        q_time = 0

        while len(chg_times_left) >= self.num_chargers:
            min_time = min(chg_times_left)
            q_time += min_time
            chg_times_left.remove(min_time)
            if pseudo_q:
                chg_times_left.append(pseudo_q.pop(0).chg_time)

        return q_time


class ChargeRequest:
    def __init__(self, time_made, veh, trip, chg_site, chg_time):
        # We can't have negative charging time.
        if chg_time < 0:
            raise ValueError('Charging time must be nonnegative, {:.5f}'
                             ' is not an acceptable value.'.format(chg_time))
        self.time_made = time_made
        self.veh = veh
        # ID is based only on vehicle
        self.id = self.veh.id
        self.trip = trip
        self.chg_site = chg_site
        self.chg_time = chg_time


class Event:
    def __init__(self, time, typ, veh, trip):
        """
        Constructor for simulation event class
        """
        # Datetime at which event occurs
        self.time = time
        # Type of event, e.g. 'trip_end'
        self.type = typ
        # Vehicle involved
        self.veh = veh
        # Trip index
        self.trip = trip


class TripStart(Event):
    def __init__(self, time, veh, trip):
        super().__init__(time, 'trip_start', veh, trip)


class TripEnd(Event):
    def __init__(self, time, veh, trip):
        super().__init__(time, 'trip_end', veh, trip)


class ChargerArrival(Event):
    def __init__(self, time, veh, trip, chg_site, chg_time):
        super().__init__(time, 'chg_arr', veh, trip)
        # Location of charger
        self.chg_site = chg_site
        # Scheduled charging time for vehicle
        self.chg_time = chg_time


class ChargerDeparture(Event):
    def __init__(self, time, veh, trip, chg_site, chg_time):
        super().__init__(time, 'chg_dpt', veh, trip)
        self.chg_site = chg_site
        self.chg_time = chg_time


class Calendar:
    def __init__(self):
        """
        Constructor for simulation calendar class
        """
        self.events = list()

    def __len__(self):
        return len(self.events)

    def sort_calendar(self):
        self.events.sort(key=operator.attrgetter('time'))

    def add_event(self, event):
        if not isinstance(event, Event):
            raise TypeError('Only objects of type Event may be added'
                            ' to calendar.')

        self.events.append(event)
        self.sort_calendar()

    def remove_event(self):
        """
        Remove first event from calendar. Call after event has been
        processed.
        """
        _ = self.events.pop(0)

    def get_next_event(self):
        return self.events[0]

    def head(self, i=5):
        i_adj = max(i, len(self.events))
        return self.events[:i_adj]


def get_object_by_id(obj_id, obj_list):
    """
    Return the object that corresponds to the given ID. Can be applied
    to Vehicle or ChargingStation (or generally, any class with .id
    field.

    :param obj_id: string giving charging object ID
    :param obj_list: list of matching objects, one of which has field
        id that matches site_id
    :return: object instance from given list
    """
    obj_matches = [o for o in obj_list if o.id == obj_id]

    if len(obj_matches) == 0:
        raise ValueError('No match for ID found in given list.')

    elif len(obj_matches) > 1:
        raise ValueError('Found {} objects in list instead of 1'
                         'with id {}'.format(len(obj_matches), obj_id))

    else:
        # We found one match, as desired
        return obj_matches[0]


class SimulationRun:
    def __init__(self, om, chg_plan, site_caps):
        """
        Constructor for simulation run class
        :param om: OperationsModel instance
        :param chg_plan: Charging schedule for all vehicles. Formatted
            as the optimal value of chg_time from an OperationsModel
        """
        MIN_CHARGE_TIME = 0.01
        self.om = om
        # Process charge plan input
        self.chg_plan = {k: chg_plan[k] if chg_plan[k] >= MIN_CHARGE_TIME
                         else 0 for k in chg_plan}
        self.vehicles = list()
        self.chargers = list()
        self.site_caps = site_caps
        self.calendar = Calendar()

        # Attributes that track outputs
        self.delay = dict()
        self.total_delay = 0
        self.queue_delay = dict()
        self.total_queue_delay = 0
        self.queue_delay_per_station = dict()
        self.rec_time = dict()
        self.total_recovery = 0
        self.unplanned_chgs = 0
        self.total_chgs = 0
        self.total_chg_time = 0
        self.pct_trips_delayed = 0

    def check_charging_needed(self, v, t, chg):
        """
        Check whether charging is needed to complete next trip
        :param v: Vehicle v under study
        :param t: Trip t under study
        :param chg: Charge level of vehicle v after trip t-1
        :return: True if charging needed, False otherwise
        """
        # Tolerance for negative charge
        eps = 1e-3

        # Calculate needed energy
        energy_rate = np.mean(list(self.om.energy_rates.values()))
        trip_dist = self.om.trip_dists[v, t]
        all_charger_dists = {s: self.om.trip_end_chg_dists[v, t, s]
                             for s in self.om.chg_sites}
        min_charger_dist = min(all_charger_dists.values())

        if t == 0 or (v, t+1) not in self.om.veh_trip_pairs:
            # Initial depot trips have artificially high distances, set
            # to 0. Return trips can charge at base, so also set to 0.
            next_dist = 0

        else:
            next_trip_dist = self.om.inter_trip_dists[v, t] + self.om.trip_dists[
                v, t+1]
            next_dist = min(min_charger_dist, next_trip_dist)

        chg_after = chg - (trip_dist + next_dist) * energy_rate

        if chg_after < -eps:
            return True
        else:
            return False

    def set_charging_plan(self, v, t, chg, avg_e_rate, buffer=0):
        """
        After determining that charging is required, choose where and
        how much to charge.

        :param v: vehicle ID
        :param t: trip number
        :param chg: current charge level
        :param avg_e_rate: expected energy consumption across trips
            in kWh/mile
        :param buffer: additional charge added beyond expected need
        :return:
        """
        # Select charger that will cause the minimum delay
        times = {s: self.om.trip_end_chg_times[v, t, s.id]
                    + self.om.trip_start_chg_times[v, t+1, s.id]
                    + s.get_queue_time()
                 for s in self.chargers}
        min_time_charger = min(times, key=times.get).id

        # Select charging amount
        req_chg = sum(avg_e_rate * (
            self.om.trip_dists[v2, t2] + self.om.inter_trip_dists[v2, t2])
            for (v2, t2) in self.om.veh_trip_pairs if v2 == v and t2 > t)
        addl_chg = req_chg + buffer - chg

        # Can't charge beyond vehicle's battery capacity
        addl_chg = min(addl_chg, self.om.chg_lims[v] - chg)

        # Update the charging plan for this vehicle so we don't charge
        # more than needed
        t_future = t + 1
        while (v, t_future) in self.om.veh_trip_pairs:
            for s in self.om.chg_sites:
                self.chg_plan[v, t_future, s] = 0
            t_future += 1

        if addl_chg < 0:
            raise ValueError('Calculated negative additional charge required')
        return min_time_charger, addl_chg

    def run_sim(self):
        # Initialize vehicles
        for v in self.om.vehicles:
            self.vehicles.append(Vehicle(id=v, max_chg=self.om.chg_lims[v]))

        for c in self.om.chg_sites:
            self.chargers.append(
                ChargingStation(id=c, power=self.om.chg_rates[c],
                                num_chargers=self.site_caps[c]))

        # Initialize calendar with first trip of all vehicles
        for v in self.vehicles:
            # Create event for first trip
            start_time = self.om.trip_start_times[v.id, 0]
            first_trip_start = TripStart(time=start_time, veh=v, trip=0)
            self.calendar.add_event(first_trip_start)

        it_ctr = 0
        # Proceed through calendar
        while len(self.calendar) > 0:
            it_ctr += 1

            # Select next event from calendar
            current_ev = self.calendar.get_next_event()

            if current_ev.type == 'trip_start':
                # Calculate delay and wait times for trip
                v = current_ev.veh
                t = current_ev.trip

                sched_start = self.om.trip_start_times[v.id, t]
                time_diff = sched_start - current_ev.time
                if time_diff >= 0:
                    # Ahead of schedule, so add to wait time
                    self.rec_time[v.id, t] = time_diff
                    self.delay[v.id, t] = 0
                else:
                    self.delay[v.id, t] = -time_diff
                    self.rec_time[v.id, t] = 0

                # Update vehicle charge
                e_rate = self.om.energy_rates[v.id, t]
                trip_dist = self.om.trip_dists[v.id, t]
                v.chg -= e_rate * trip_dist

                # Complete trip and add its end time to calendar
                sched_end = self.om.trip_end_times[v.id, t]
                actual_end = sched_end + self.delay[v.id, t]
                new_ev = TripEnd(time=actual_end, veh=v, trip=t)
                self.calendar.add_event(new_ev)

            elif current_ev.type == 'trip_end':
                v = current_ev.veh
                t = current_ev.trip

                # Are any trips left on the schedule? If not, return to
                # base and do not add any new events.
                if (v.id, t+1) not in self.om.veh_trip_pairs:
                    pass

                else:
                    # If a charge is scheduled at this time, drive to
                    # the charger.
                    try:
                        charges = {s: self.chg_plan[v.id, t, s]
                                   for s in self.om.chg_sites}
                        max_chg_val = max(charges.values())
                    except KeyError:
                        # This KeyError may be raised if (1) we are on
                        # trip zero for some vehicle, which never has
                        # charging because it's just driving from the
                        # depot or (2) the key was not provided because
                        # the charging plan input only contains blocks
                        # that are expected to require charging. So we
                        # specify manually that no charging is planned
                        # in either case.
                        max_chg_val = 0

                    if max_chg_val > 0:
                        # Charging is scheduled for after this trip.
                        # Identify charging site, drive time, and
                        # charge required to drive there.
                        chg_site = max(charges, key=charges.get)
                        time_to_site = self.om.trip_end_chg_times[v.id, t,
                                                                  chg_site]
                        charger_arrival_time = current_ev.time + time_to_site
                        chg_used = self.om.energy_rates[v.id, t] \
                            * self.om.trip_end_chg_dists[v.id, t, chg_site]
                        v.chg -= chg_used
                        new_ev = ChargerArrival(
                            time=charger_arrival_time, veh=v, trip=t,
                            chg_time=self.chg_plan[v.id, t, chg_site],
                            chg_site=chg_site)
                        self.calendar.add_event(new_ev)

                    else:
                        # Charging is not scheduled after this trip.
                        # Check if it needs to be done to complete the
                        # next trip.
                        chg_needed = self.check_charging_needed(v.id, t+1,
                                                                v.chg)
                        if chg_needed:
                            self.unplanned_chgs += 1
                            # Charging must be done. Choose where and
                            # how much.
                            avg_energy_rate = np.mean(
                                list(self.om.energy_rates.values()))
                            chg_site, chg_amt = self.set_charging_plan(
                                v.id, t, v.chg, avg_energy_rate)
                            chg_amt = chg_amt*1.1
                            chg_time = chg_amt / self.om.chg_rates[chg_site]
                            time_to_site = self.om.trip_end_chg_times[v.id, t,
                                                                      chg_site]
                            charger_arrival_time = current_ev.time \
                                + time_to_site
                            chg_used = self.om.energy_rates[v.id, t] \
                                * self.om.trip_end_chg_dists[v.id, t, chg_site]
                            v.chg -= chg_used
                            new_ev = ChargerArrival(
                                time=charger_arrival_time, veh=v, trip=t,
                                chg_site=chg_site, chg_time=chg_time)
                            self.calendar.add_event(new_ev)

                        else:
                            # No need to charge. Move to next trip.
                            time_to_next = self.om.inter_trip_times[v.id, t]
                            next_start_time = current_ev.time + time_to_next
                            next_start_dist = self.om.inter_trip_dists[v.id, t]
                            e_rate = self.om.energy_rates[v.id, t]
                            v.chg -= next_start_dist * e_rate
                            new_ev = TripStart(
                                time=next_start_time, veh=v, trip=t+1)
                            self.calendar.add_event(new_ev)

            elif current_ev.type == 'chg_arr':
                chg_site = get_object_by_id(current_ev.chg_site, self.chargers)
                v = current_ev.veh
                chg_req = ChargeRequest(
                    time_made=current_ev.time, veh=v, trip=current_ev.trip,
                    chg_site=chg_site, chg_time=current_ev.chg_time)

                if chg_site.is_full():
                    # If charger is full, add the arriving vehicle to
                    # the queue
                    chg_site.add_to_queue(chg_req)

                else:
                    # If charger is available, start charging and add
                    # event for charge completion to calendar.
                    chg_site.start_charging(chg_req)
                    chg_end_time = current_ev.time + current_ev.chg_time
                    new_ev = ChargerDeparture(
                        time=chg_end_time, veh=v, trip=current_ev.trip,
                        chg_site=current_ev.chg_site,
                        chg_time=current_ev.chg_time)
                    # No queue delay is incurred because this vehicle
                    # never enters a station queue
                    self.queue_delay[
                        v.id, current_ev.trip, current_ev.chg_site] = 0
                    self.calendar.add_event(new_ev)

            elif current_ev.type == 'chg_dpt':
                v = current_ev.veh
                t = current_ev.trip
                s = current_ev.chg_site

                chg_site = get_object_by_id(s, self.chargers)

                try:
                    chg_site.finish_charging(v.id)
                except ValueError:
                    raise ValueError('Cannot remove vehicle that is not'
                                     'charging.')

                # Track outputs
                self.total_chgs += 1
                self.total_chg_time += current_ev.chg_time

                # Advance queue. If a new vehicle starts charging, we
                # need to create an event for its charge completion.
                new_req = chg_site.advance_queue()
                if new_req is not None:
                    # Track how long this request waited for
                    self.queue_delay[new_req.veh.id, new_req.trip, s] = \
                        current_ev.time - new_req.time_made
                    # Process new charging request, add departure to
                    # calendar.
                    req_chg_end = current_ev.time + new_req.chg_time
                    new_dept = ChargerDeparture(
                        time=req_chg_end, veh=new_req.veh, trip=new_req.trip,
                        chg_site=s, chg_time=new_req.chg_time)
                    self.calendar.add_event(new_dept)

                # Update battery level
                v.chg += current_ev.chg_time * chg_site.power

                # Move vehicle to start of next trip
                dist_to_start = self.om.trip_start_chg_dists[v.id, t+1, s]
                time_to_start = self.om.trip_start_chg_times[v.id, t+1, s]
                v.chg -= dist_to_start * self.om.energy_rates[v.id, t]
                time_at_start = current_ev.time + time_to_start
                new_ev = TripStart(time=time_at_start, veh=v, trip=t+1)
                self.calendar.add_event(new_ev)

            else:
                raise AttributeError('Unrecognized or absent event type:'
                                     '{}'.format(current_ev.type))

            # Remove completed event from calendar
            self.calendar.remove_event()

    def process_results(self):
        # Total charging time

        # Total delay
        self.total_delay = sum(self.delay.values())

        # Total queue delay
        self.total_queue_delay = sum(self.queue_delay.values())

        # Queue delay per station
        # Results for each queue
        q_sites = set([s for (v, t, s) in self.queue_delay])
        for s in q_sites:
            q_vals = {(v, t): self.queue_delay[v, t, s2]
                      for (v, t, s2) in self.queue_delay if s2 == s}
            times_in_q = np.array(list(q_vals.values()))
            self.queue_delay_per_station[s] = times_in_q.tolist()
        #     plt.figure()
        #     plt.hist(times_in_q)
        #     plt.title('Times in Queue for {}'.format(s))
        #     plt.xlabel('Time (min)')
        # plt.show()

        # % trips delayed
        all_delays = [self.delay[v, t] for (v, t) in self.om.charging_vts]
        self.pct_trips_delayed = len([d for d in all_delays if d > 5]) / len(
            all_delays) * 100

        # Total recovery time
        self.total_recovery = sum(self.rec_time.values())

    def print_results(self):
        print('Total recovery time: {:.2f} minutes'.format(
            self.total_recovery))
        print('Total delay: {:.2f} minutes'.format(self.total_delay))
        print('Maximum delay: {:.2f} minutes'.format(max(self.delay.values())))
        # print('Average delay per trip: {:.2f} minutes'.format(
        #     self.total_delay / len(self.om.veh_trip_pairs)))
        print('Percentage of trips delayed over 5 minutes: {:.2f}%'.format(
            self.pct_trips_delayed))
        print('Total queue waiting time: {:.2f} minutes'.format(
            sum(self.queue_delay.values())))
        print('Total number of charger visits: {}'.format(self.total_chgs))
        print('Number of unscheduled charges: {}'.format(self.unplanned_chgs))
        print('Total charging time: {:.2f} minutes'.format(
            self.total_chg_time))
        print('Average time per charge: {:.2f} minutes'.format(
            self.total_chg_time / self.total_chgs))

        q_sites = set([s for (v, t, s) in self.queue_delay])
        for s in q_sites:
            print('Mean time in queue for {}: {} minutes'.format(s,
                  np.mean(self.queue_delay_per_station[s])))


class SimulationBatch:
    def __init__(
            self, om, sched, n_sims, site_caps, energy_std=0.1, seed=None):
        # Perform batch simulation run. As of now, only energy usage
        # rate is assumed to vary; all other parameters are known.
        self.om = om
        self.sched = sched
        self.n_sims = n_sims
        self.site_caps = site_caps
        self.delay = np.zeros(n_sims)
        self.energy_std = energy_std
        np.random.seed(seed)

    def run(self):
        for n in range(self.n_sims):
            self.om.energy_rates = {
                k: max(np.random.normal(
                    np.mean(list(self.om.energy_rates.values())),
                    self.energy_std), 0)
                for k in self.om.energy_rates}
            sim = SimulationRun(self.om, self.sched, self.site_caps)
            sim.run_sim()
            sim.process_results()
            # sim.print_results()
            self.delay[n] = sim.total_delay

    def process_results(self):
        delay_mean = np.mean(self.delay)
        print('Mean delay: {:.2f} minutes'.format(delay_mean))
        delay_std = np.std(self.delay, ddof=1)
        print('Standard deviation: {:.2f} minutes'.format(delay_std))
        # Calculate 95% confidence interval
        alpha = 0.05
        t_val = tstat.ppf(1-alpha/2, self.n_sims-1)
        half_len = t_val*delay_std/np.sqrt(self.n_sims)
        print('95% confidence interval on mean delay: [{:.2f}, {:.2f}]'.format(
            delay_mean - half_len, delay_mean + half_len))

    def plot_delay_hist(self):
        plt.hist(self.delay)
        plt.title('Histogram of Delay for {} Simulations'.format(self.n_sims))
        plt.show()


if __name__ == '__main__':
    import pickle

    with open('flm_new_queue.pickle', 'rb') as f:
        op = pickle.load(f)

    print('OPTIMIZATION RESULTS')
    op.print_results()
    print()

    # Extract needed outputs
    sched = op.chg_schedule
    # chg_sites = op.opt_stations
    op.veh_trip_pairs = op.charging_vts
    site_cap = {s: 10 for s in op.chg_sites}
    batch = SimulationBatch(om=op, sched=sched, n_sims=1, site_caps=site_cap,
                            energy_std=0.01, seed=42)
    print('SIMULATION RESULTS')
    batch.run()
    batch.process_results()
    batch.plot_delay_hist()





