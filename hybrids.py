import numpy as np
import pandas as pd
import numba
from numba import prange
import requests
import os
import json
import time
from io import StringIO


@numba.njit
def find_least_cost_option(configuration, temp, ghi, hour_numbers, load_curve, inv_eff, n_dis, n_chg, dod_max,
                           diesel_price, end_year, start_year, pv_cost, charge_controller, pv_inverter,
                           pv_om, diesel_cost, diesel_om, battery_inverter_life, battery_inverter_cost, diesel_life,
                           pv_life, battery_cost, discount_rate, lpsp_max, diesel_limit, full_life_cycles):
    pv = float(configuration[0])
    battery = float(configuration[1])
    usable_battery = battery * dod_max  # ensure the battery never goes below max depth of discharge
    diesel = float(configuration[2])

    annual_demand = load_curve.sum()

    # First the PV generation and net load (load - pv generation) is calculated for each hour of the year
    net_load, pv_gen = pv_generation(temp, ghi, pv, load_curve, inv_eff)

    # For each hour of the year, diesel generation, battery charge/discharge and performance variables are calculated.
    diesel_generation_share, battery_life, unmet_demand_share, annual_fuel_consumption, \
        excess_gen_share, battery_soc_curve, diesel_gen_curve = \
        year_simulation(battery_size=usable_battery, diesel_capacity=diesel, net_load=net_load,
                        hour_numbers=hour_numbers, inv_eff=inv_eff, n_dis=n_dis, n_chg=n_chg,
                        annual_demand=annual_demand, full_life_cycles=full_life_cycles, dod_max=dod_max)

    # If the system could meet the demand in a satisfactory manner (i.e. with high enough reliability and low enough
    # share of the generation coming from the diesel generator), then the LCOE is calculated. Else 99 is returned.
    if (battery_life == 0) or (unmet_demand_share > lpsp_max) or (diesel_generation_share > diesel_limit):
        lcoe = 99
        investment = 0
        fuel_cost = 0
        om_cost = 0
        npc = 0
    else:
        lcoe, investment, battery_investment, fuel_cost, \
            om_cost, npc, fuel_usage, annual_om = calculate_hybrid_lcoe(diesel_price=diesel_price,
                                                 end_year=end_year,
                                                 start_year=start_year,
                                                 annual_demand=annual_demand,
                                                 fuel_usage=annual_fuel_consumption,
                                                 pv_panel_size=pv,
                                                 pv_cost=pv_cost,
                                                 charge_controller=charge_controller,
                                                 pv_inverter_cost=pv_inverter,
                                                 pv_om=pv_om,
                                                 diesel_capacity=diesel,
                                                 diesel_cost=diesel_cost,
                                                 diesel_om=diesel_om,
                                                 battery_inverter_cost=battery_inverter_cost,
                                                 battery_inverter_life=battery_inverter_life,
                                                 load_curve=load_curve,
                                                 diesel_life=diesel_life,
                                                 pv_life=pv_life,
                                                 battery_life=battery_life,
                                                 battery_size=battery,
                                                 battery_cost=battery_cost,
                                                 discount_rate=discount_rate)

    return lcoe, unmet_demand_share, diesel_generation_share, investment, fuel_cost, om_cost, battery, \
        battery_life, pv, diesel, npc, fuel_usage, annual_om

@numba.njit
def pv_generation(temp, ghi, pv_capacity, load, inv_eff):
    # Calculation of PV gen and net load
    k_t = 0.005  # temperature factor of PV panels
    t_cell = temp + 0.0256 * ghi  # PV cell temperature
    pv_gen = pv_capacity * 0.9 * ghi / 1000 * (1 - k_t * (t_cell - 25))  # PV generation in the hour
    net_load = load - pv_gen * inv_eff  # remaining load not met by PV panels
    return net_load, pv_gen

@numba.njit
def year_simulation(battery_size, diesel_capacity, net_load, hour_numbers, inv_eff, n_dis, n_chg,
                    annual_demand, full_life_cycles, dod_max):
    soc = 0.5  # Initial SOC of battery

    # Variables for tracking annual performance information
    annual_unmet_demand = 0
    annual_excess_gen = 0
    annual_diesel_gen = 0
    annual_battery_use = 0
    annual_fuel_consumption = 0

    # Arrays for tracking hourly values throughout the year (for plotting purposes)
    diesel_gen_curve = []
    battery_soc_curve = []

    # Run the simulation for each hour during one year
    for hour in hour_numbers:
        load = net_load[int(hour)]

        diesel_gen, annual_fuel_consumption, annual_diesel_gen, annual_battery_use, soc, annual_unmet_demand, \
            annual_excess_gen = hour_simulation(hour, soc, load, diesel_capacity, annual_fuel_consumption,
                                                annual_diesel_gen,
                                                inv_eff, n_dis, n_chg, battery_size, annual_battery_use,
                                                annual_unmet_demand,
                                                annual_excess_gen)

        # Update plotting arrays
        diesel_gen_curve.append(diesel_gen)
        battery_soc_curve.append(soc)

    # When a full year has been simulated, calculate battery life and performance metrics
    if (battery_size > 0) & (annual_battery_use > 0):
        battery_life = min(round(full_life_cycles / (annual_battery_use)), 20)  # ToDo should dod_max be included here?
    else:
        battery_life = 20

    unmet_demand_share = annual_unmet_demand / annual_demand  # LPSP is calculated
    excess_gen_share = annual_excess_gen / annual_demand
    diesel_generation_share = annual_diesel_gen / annual_demand

    return diesel_generation_share, battery_life, unmet_demand_share, annual_fuel_consumption, \
        excess_gen_share, battery_soc_curve, diesel_gen_curve


@numba.njit
def hour_simulation(hour, soc, net_load, diesel_capacity, annual_fuel_consumption, annual_diesel_gen, inv_eff, n_dis,
                    n_chg, battery_size, annual_battery_use, annual_unmet_demand, annual_excess_gen):
    # First the battery self-discharge is calculated (default rate set to 0.02% of the state-of-charge - SOC - per hour)
    battery_use = 0.0002 * soc
    soc - 0.0002 * soc

    battery_dispatchable = soc * battery_size * n_dis * inv_eff  # Max load that can be met by the battery until empty
    battery_chargeable = (1 - soc) * battery_size / n_chg / inv_eff  # Max energy that can be used to charge the battery until full

    # Below is the dispatch strategy for the diesel generator and battery

    if 4 < hour <= 17:
        # During the morning and day, the batteries are dispatched primarily.
        # The diesel generator, if needed, is run at the lowest possible capacity
        # (it is assumed that the diesel generator should never run below 40% of rated capacity)

        # Minimum diesel capacity to cover the net load after batteries.
        # Diesel generator limited by lowest possible capacity (40%) and rated capacity

        if net_load < battery_dispatchable:  # If load can be met by batteries, diesel generator is not used
            diesel_gen = 0
        else:  # If batteries are insufficient, diesel generator is used
            diesel_gen = min(max(net_load - battery_dispatchable, 0.4 * diesel_capacity), diesel_capacity)

    elif 17 < hour <= 23:
        # During the evening, the diesel generator is dispatched primarily.
        # During this time, diesel generator seeks to meet load and charge batteries for the night if possible.
        # Batteries are dispatched if diesel generation is insufficient.

        # Maximum amount of diesel needed to supply load and charge battery
        # Diesel generator limited by lowest possible capacity (40%) and rated capacity
        max_diesel = max(min(net_load + battery_chargeable, diesel_capacity), 0.4 * diesel_capacity)

        if net_load > 0:  # If there is net load after PV generation, diesel generator is used
            diesel_gen = max_diesel
        else:  # Else if there is no remaining load, diesel generator is not used
            diesel_gen = 0

    else:
        # During night, batteries are dispatched primarily. If batteries are insufficient, the diesel generator is used
        # at highest capacity required to meet load and charge batteries in order to minimize operating hours at night

        # Maximum amount of diesel needed to supply load and charge battery
        # Diesel generator limited by lowest possible capacity (40%) and rated capacity

        if net_load < battery_dispatchable:  # If load can be met by batteries, diesel generator is not used
            diesel_gen = 0
        else:  # If batteries are insufficient, diesel generator is used
            diesel_gen = max(min(net_load + battery_chargeable, diesel_capacity), 0.4 * diesel_capacity)

    if diesel_gen > 0:  # If the diesel generator was used, the amount of diesel generation and fuel used is stored
        annual_fuel_consumption += diesel_capacity * 0.08145 + diesel_gen * 0.246
        annual_diesel_gen += diesel_gen

    net_load -= diesel_gen  # Remaining load after diesel generation, used for subsequent battery calculations etc.

    soc_prev = soc  # Store the battery SOC before the hour in a variable to ensure battery is not over-used

    if (net_load > 0) & (battery_size > 0):
        if diesel_gen > 0:
            # If diesel generation is used, but is smaller than load, battery is discharged
            soc -= net_load / n_dis / inv_eff / battery_size
        elif diesel_gen == 0:
            # If net load is positive and no diesel is used, battery is discharged
            soc -= net_load / n_dis / inv_eff / battery_size
    elif (net_load < 0) & (battery_size > 0):
        if diesel_gen > 0:
            # If diesel generation is used, and is larger than load, battery is charged
            soc -= net_load * n_chg * inv_eff / battery_size
        if diesel_gen == 0:
            # If net load is negative, and no diesel has been used, excess PV gen is used to charge battery
            soc -= net_load * n_chg / battery_size

    # Store how much battery energy (measured in SOC) was discharged (if used).
    # No more than the previous SOC can be used
    if (net_load > 0) & (battery_size > 0):
        battery_use += min(net_load / n_dis / battery_size, soc_prev)

    annual_battery_use += battery_use

    # Calculate if there was any unmet demand or excess energy generation during the hour

    if battery_size > 0:
        if soc < 0:
            # If State of charge is negative, that means there's demand that could not be met.
            # If so, the annual unmet demand variable is updated and the SOC is reset to empty (0)
            annual_unmet_demand -= soc / n_dis * battery_size
            soc = 0

        if soc > 1:
            # If State of Charge is larger than 1, that means there was excess PV/diesel generation
            # If so, the annual excess generation variable is updated and the SOC is set to full (1)
            annual_excess_gen += (soc - 1) / n_chg * battery_size
            soc = 1
    else:  # This part handles the same case, if no battery is included in the system
        if net_load > 0:
            annual_unmet_demand += net_load
        if net_load < 0:
            annual_excess_gen -= net_load

    return diesel_gen, annual_fuel_consumption, annual_diesel_gen, annual_battery_use, \
        soc, annual_unmet_demand, annual_excess_gen


@numba.njit
def calculate_hybrid_lcoe(diesel_price, end_year, start_year, annual_demand,
                          fuel_usage, pv_panel_size, pv_cost, pv_life, pv_om, charge_controller, pv_inverter_cost,
                          diesel_capacity, diesel_cost, diesel_om, diesel_life,
                          battery_size, battery_cost, battery_life, battery_inverter_cost, battery_inverter_life,
                          load_curve, discount_rate):

    # Necessary information for calculation of LCOE is defined
    project_life = end_year - start_year  # Calculate project lifetime
    generation = np.ones(project_life) * annual_demand  # array of annual demand
    generation[0] = 0  # In first year, there is assumed to be no generation

    # Calculate LCOE
    sum_el_gen = 0
    investment = 0
    sum_costs = 0
    total_battery_investment = 0
    total_fuel_cost = 0
    total_om_cost = 0
    npc = 0

    # Iterate over each year in the project life and account for the costs that incur
    # This includes investment, OM, fuel, and reinvestment in any year a technology lifetime expires
    for year in prange(project_life + 1):
        salvage = 0
        inverter_investment = 0
        diesel_investment = 0
        pv_investment = 0
        battery_investment = 0

        fuel_costs = fuel_usage * diesel_price
        om_costs = (pv_panel_size * (pv_cost + charge_controller) * pv_om + diesel_capacity * diesel_cost * diesel_om)

        total_fuel_cost += fuel_costs / (1 + discount_rate) ** year
        total_om_cost += om_costs / (1 + discount_rate) ** year

        # Here we check if there is need for investment/reinvestment
        if year % battery_inverter_life == 0:
            inverter_investment = max(load_curve) * battery_inverter_cost  # Battery inverter, sized based on the peak demand in the year
        if year % diesel_life == 0:
            diesel_investment = diesel_capacity * diesel_cost
        if year % pv_life == 0:
            pv_investment = pv_panel_size * (pv_cost + charge_controller + pv_inverter_cost)  # PV inverter and charge controller are sized based on the PV panel rated capacity
        if year % battery_life == 0:
            battery_investment = battery_size * battery_cost

        # In the final year, the salvage value of all components is calculated based on remaining life
        if year == project_life:
            salvage = (1 - (project_life % battery_life) / battery_life) * battery_cost * battery_size + \
                      (1 - (project_life % diesel_life) / diesel_life) * diesel_capacity * diesel_cost + \
                      (1 - (project_life % pv_life) / pv_life) * pv_panel_size * (
                              pv_cost + charge_controller + pv_inverter_cost) + \
                      (1 - (project_life % battery_inverter_life) / battery_inverter_life) * max(
                load_curve) * battery_inverter_cost

            total_battery_investment -= (1 - (
                    project_life % battery_life) / battery_life) * battery_cost * battery_size

        investment += (diesel_investment + pv_investment + battery_investment + inverter_investment) / ((1 + discount_rate) ** year) # Removed salvage
        total_battery_investment += battery_investment

        sum_costs += (fuel_costs + om_costs + battery_investment + diesel_investment + pv_investment +
                      inverter_investment - salvage) / ((1 + discount_rate) ** year)

        npc += (fuel_costs + om_costs + battery_investment + diesel_investment + pv_investment +
                inverter_investment) / ((1 + discount_rate) ** year)

        if year > 0:
            sum_el_gen += annual_demand / ((1 + discount_rate) ** year)

    return sum_costs / sum_el_gen, investment, total_battery_investment, total_fuel_cost, total_om_cost, npc, fuel_usage, om_costs + fuel_costs


@numba.njit
def calc_load_curve(bba, bbb, bbc, bbpme, bbprod):
    # the values below define the load curve for the five tiers. The values reflect the share of the daily demand
    # expected in each hour of the day (sum of all values for one tier = 1)
    ba_load_curve = [0.988074348, 0.83234137, 0.780348447, 0.895071886, 1.29452897, 1.427317285,
                     1.427143895, 2.036533853, 2.788021073, 3.132621618, 3.398067297, 3.597146625,
                     3.749088177, 3.619622164, 3.645379008, 3.798564636, 4.140905488, 6.288317305,
                     9.012611343, 8.343100334, 6.300316324, 3.931731497, 2.193525721, 1.379620337]
    
    bb_load_curve = [3.184482155, 2.873016199, 2.769030352, 2.99847723, 3.797391399, 4.062968029, 
                     4.062621248, 5.281401164, 6.784375605, 7.473576694, 8.004468052, 8.402626707, 
                     8.706509811, 8.447577786, 8.499091475, 8.805462731, 9.490144434, 13.78496807, 
                     19.23355614, 17.89453413, 13.80896611, 9.071796452, 5.5953849, 3.967574132]
    
    bc_load_curve = [8.923873354, 8.047396638, 7.614331638, 7.763438133, 9.507818801, 12.4321089,
                     13.48321774, 14.41446325, 15.87567906, 16.28469545, 17.16546017, 17.7633564,
                     18.49249621, 19.22393746, 19.1473699, 18.37923754, 17.41429094, 23.6968058, 
                     40.54912477, 48.55186439, 44.0975794, 32.05947574, 19.32369079, 11.78828855]
    
    bpme_load_curve = [0.250823098, 0.250823098, 0.250823098, 0.250823098, 0.250823098, 0.250823098,
                       7.645880133, 18.30034088, 30.36078872, 42.83318642, 49.9405017, 52.93860082,
                       49.93452555, 44.38491667, 35.85872985, 27.40653484, 20.76932651, 32.56118326,
                       62.28781859, 74.50081552, 62.8359663, 36.32711164, 17.17269706, 4.690252421]
    
    bprod_load_curve = [12.51341557,  12.49088602, 12.47672647, 16.65231446, 21.62811274, 73.12221043, 
                        307.3487742, 619.3246364, 907.5906364, 919.5741762, 1038.177627, 1124.612654,
                        1022.608964, 896.7845886, 959.4059058, 857.2820323, 729.2917592, 482.4635076,
                        108.3151878, 50.78451089, 60.39546483, 26.100591, 13.54589784, 12.509420]
    
    ba_load_curve = [i * bba for i in ba_load_curve]
    bb_load_curve = [i * bbb for i in bb_load_curve]
    bc_load_curve = [i * bbc for i in bc_load_curve]
    bpme_load_curve = [i * bbpme for i in bpme_load_curve]
    bprod_load_curve = [i * bbprod for i in bprod_load_curve]
    
    load_curve = [sum(x) for x in zip(ba_load_curve, bb_load_curve, bc_load_curve, bpme_load_curve, bprod_load_curve )] * 365
    load_curve = [i / 1000 for i in load_curve]
    
    return np.array(load_curve) 
    
#load_curve = load_curves(bba, bbb, bbc, bbpme, bbprod)
#energy_per_hh = (sum(load_curves(bba, bbb, bbc, bbpme, bbprod)))


def get_pv_data(latitude, longitude, token, output_folder):
    # This function can be used to retrieve solar resource data from https://renewables.ninja
    api_base = 'https://www.renewables.ninja/api/'
    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}

    out_path = os.path.join(output_folder, 'pv_data_lat_{}_long_{}.csv'.format(latitude, longitude))

    url = api_base + 'data/pv'

    args = {
        'lat': latitude,
        'lon': longitude,
        'date_from': '2020-01-01',
        'date_to': '2020-12-31',
        'dataset': 'merra2',
        'capacity': 1.0,
        'system_loss': 0.1,
        'tracking': 0,
        'tilt': 35,
        'azim': 180,
        'format': 'json',
        'local_time': True,
        'raw': True
    }

    if token != '':

        try:
            r = s.get(url, params=args)

            # Parse JSON to get a pandas.DataFrame of data and dict of metadata
            parsed_response = json.loads(r.text)

        except json.decoder.JSONDecodeError:
            print('API maximum hourly requests reached, waiting one hour', time.ctime())
            time.sleep(3700)
            print('Wait over, resuming API requests', time.ctime())
            r = s.get(url, params=args)

            # Parse JSON to get a pandas.DataFrame of data and dict of metadata
            parsed_response = json.loads(r.text)

        data = pd.read_json(StringIO(json.dumps(parsed_response['data'])), orient='index')

        df_out = pd.DataFrame(columns=['time', 'ghi', 'temp'])
        df_out['ghi'] = (data['irradiance_direct'] + data['irradiance_diffuse']) * 1000
        df_out['temp'] = data['temperature']
        df_out['time'] = data['local_time']

        df_out.to_csv(out_path, index=False)
        
    else:
        print('No token provided')


def read_environmental_data(path, skiprows=24, skipcols=1):
    """
    This method reads the solar resource GHI and temperature for each hour during one year from a csv-file.
    The skiprows and skipcolumns define which rows and columns the data should be read from.
    """
    try:
        data = pd.read_csv(path, skiprows=skiprows)
        ghi_curve = data.iloc[:, 0 + skipcols].values
        temp = data.iloc[:, 1 + skipcols].values

        return ghi_curve, temp
    except:
        print('Could not read data, try changing which columns and rows ro read')

def calculate_distribution_lcoe(end_year, start_year, annual_demand,
                                distribution_cost, om_costs, distribution_life,
                                discount_rate):
    # Calculate project lifetime
    project_life = end_year - start_year
    
    # Create an array of annual demand (generation)
    generation = np.ones(project_life) * annual_demand
    generation[0] = 0  # Assume no generation in the first year
    
    # Initialize cost variables
    sum_el_gen = 0
    investment = 0
    sum_costs = 0
    total_om_cost = 0
    npc = 0

    # Iterate over each year in the project life
    for year in prange(project_life + 1):
        salvage = 0
        distribution_investment = 0

        # Accumulate total O&M costs discounted
        total_om_cost += om_costs / ((1 + discount_rate) ** year)
        
        # Reinvestment in distribution grid based on its lifetime
        if year % distribution_life == 0:
            distribution_investment = distribution_cost
        
        # Calculate salvage value in the final year
        if year == project_life:
            salvage = (1 - (project_life % distribution_life) / distribution_life) * distribution_cost
        
        # Accumulate investment, accounting for salvage in the final year
        investment += (distribution_investment) / ((1 + discount_rate) ** year) # Removed salvage

        # Accumulate total costs discounted
        sum_costs += (om_costs + distribution_investment - salvage) / ((1 + discount_rate) ** year)

        # Accumulate net present cost (NPC)
        npc += (om_costs + distribution_investment) / ((1 + discount_rate) ** year)
        
        # Accumulate total electricity generation discounted
        if year > 0:
            sum_el_gen += annual_demand / ((1 + discount_rate) ** year)

    # Calculate LCOE
    lcoe = sum_costs / sum_el_gen
    
    return lcoe, investment, total_om_cost, npc, sum_el_gen, om_costs

