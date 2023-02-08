# Optimal Charging Infrastructure Design for Battery-Electric Buses
This repository hosts models and data used to determine optimal charging station locations for battery-electric buses, applied to a case study of King County, WA. A full description of the models used and the case study application can be found in Dan McCabe's M.S. thesis from the University of Washington, available at https://digital.lib.washington.edu/researchworks/handle/1773/47413.

# Acknowledgement
This work is supported by the National Science Foundation (NSF) Graduate Research Fellowship Program under Grant No. DGE-1762114. This research is also partially supported by a grant from the Pacific Northwest Transportation Consortium (PacTrans) funded by the US Department of Transportation (USDOT).

# Code Structure
Below is the schema of the full repo:
- beb_chargers
    - data
        - gtfs
        - so_king_cty_sites.csv
    - beb_data.py
    - beb_model.py
    - beb_vis.py
    - evaluation.py
    - queue_lambda_sensitivity.py
    - so_king_county_study.py

The `data` folder contains data files used for the project. `so_king_cty_sites.csv` gives the names and coordinates of the seven candidate charging sites considered for the case study. The gtfs `folder` contains GTFS data files pulled from the King County Metro GTFS feed in spring 2019. For details on the GTFS format, see the documentation at https://developers.google.com/transit/gtfs/reference.

`beb_data` contains functions used for pulling and processing relevant data, including GTFS. `beb_model` contains code for the optimization models developed for the project. `beb_vis` contains visualization code. `evaluation` includes a discrete-event simulation model used to evaluate the true performance of both models, including actual queue times and trip delays. `queue_lambda_sensitivity` runs a sensitivity analysis for the linear queue model parameter $\lambda$. `so_king_county_study` is a script that can be used to run the models and evaluate the results using the simulation.