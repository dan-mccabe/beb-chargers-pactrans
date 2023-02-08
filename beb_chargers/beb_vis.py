import mplleaflet
import matplotlib.pyplot as plt


def get_shape(shapes_df, shape_id):
    this_shape = shapes_df[shapes_df['shape_id'] == shape_id]
    this_shape.sort_values(by='shape_pt_sequence')
    return this_shape['shape_pt_lon'], this_shape['shape_pt_lat']


def plot_trips(all_trips_df, shapes_df):
    trips_per_shape = all_trips_df.groupby(
        'shape_id').count().sort_values(by='route_id')['route_id']
    line_alphas = {s: max(trips_per_shape[s] / max(trips_per_shape), 0.1)
                   for s in trips_per_shape.index}
    fig = plt.figure()
    for s in all_trips_df['shape_id'].unique():
        lon, lat = get_shape(shapes_df, s)
        plt.plot(lon, lat, color='b', alpha=line_alphas[s], linewidth=3)
    mplleaflet.show(fig)


def plot_results(results_df, built_stations_df, **kwargs):
    unique_chg_ends = results_df['term_lat_lon'].unique()
    unique_chg_x = [i[1] for i in unique_chg_ends]
    unique_chg_y = [i[0] for i in unique_chg_ends]

    # Plot all unique terminal nodes for charging and their
    # corresponding charging stations
    fig = plt.figure()
    for ind, row in results_df.iterrows():
        x = [row['term_lat_lon'][1], row['charger_lat_lon'][1]]
        y = [row['term_lat_lon'][0], row['charger_lat_lon'][0]]
        plt.plot(x, y, alpha=0.4, color='k', linewidth=5)
    plt.scatter(unique_chg_x, unique_chg_y, s=100, color='b', alpha=0.5)
    plt.scatter(built_stations_df['x'], built_stations_df['y'], s=300,
                color='g', alpha=0.5)
    mplleaflet.show(fig, **kwargs)


def plot_terminal_nodes(all_trips_df, sites_df, **kwargs):
    term_counts = all_trips_df.groupby(
        'term_lat_lon').count()['trip_sequence'].sort_values().reset_index()
    fig = plt.figure()
    for n in term_counts.index:
        size = max(1.2*term_counts[n], 50)
        plt.scatter(n[1], n[0], s=size, alpha=0.5,
                    color='b', **kwargs)

    sites_df = sites_df.reset_index()
    plt.scatter(sites_df['x'], sites_df['y'], s=200, color='g', alpha=0.5)
    for i in sites_df.index:
        plt.annotate(str(i+1), (sites_df.at[i, 'x'], sites_df.at[i, 'y']))

    mplleaflet.show(fig)


def get_node_legend():
    fig = plt.figure()
    endpt = plt.scatter(0, 0, s=100, color='b', alpha=0.5)
    cand = plt.scatter(0, 0, s=100, color='g', alpha=0.5)
    fig.legend([endpt, cand], ['Trip end', 'Candidate charger site'],
               loc='upper center')
    plt.savefig('nodes_legend.png', dpi=1200)



