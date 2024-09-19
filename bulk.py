import numpy as np
import pandas as pd
import os
import time
from shapely.geometry import Point
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.optimize import Bounds, differential_evolution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from hybrids import calc_load_curve, find_least_cost_option, get_pv_data

def centroids_to_extract_solar_data(admin0, target_crs):
    """
    This function generates a grid of evenly spaced centroid points within the bounding box of the given 
    administrative area's geometry ('admin0'). It then clips these points to ensure they fall within 
    the boundaries of 'admin0' and converts the points to WGS 84 (EPSG:4326) for further solar data extraction.
    
    Parameters:
    admin0 (GeoDataFrame): The input GeoDataFrame containing the administrative boundary geometry.

    Returns:
    GeoDataFrame: A GeoDataFrame of centroid points clipped to the 'admin0' boundaries, reprojected to WGS 84.
    """
    
    # Extract geometry and calculate bounding box
    admin_geometry = admin0["geometry"]
    xmin, ymin, xmax, ymax = admin_geometry.total_bounds
    len_x = xmax - xmin
    len_y = ymax - ymin
    
    # Define the grid size (7x7) and calculate step sizes for centroids
    n_x = 7
    n_y = 7
    delta_x = len_x / n_x
    delta_y = len_y / n_y
    
    # Initialize variables for storing centroids and their IDs
    points_for_ghi = []
    id_list = []
    id_generator = 0
    xupdated = xmin
    
    # Generate centroids across the grid within the bounding box
    while xupdated < xmax:
        yupdated = ymin  # Reset y position for each x iteration
        while yupdated < ymax:
            points_xy = Point(xupdated, yupdated)
            id_list.append(id_generator)
            id_generator += 1
            points_for_ghi.append(points_xy)
            yupdated += delta_y  # Move to the next y position
        xupdated += delta_x  # Move to the next x position
    
    # Create a GeoDataFrame for the generated points, assigning the given CRS
    gpd_points = gpd.GeoDataFrame(
        {'id': id_list, 'geometry': points_for_ghi},
        crs=target_crs
    )
    
    # Clip the points to ensure they are within the boundaries of the admin0 geometry
    clipped_points = gpd.clip(gpd_points, admin0)
    
    # Reproject the points to WGS 84 (EPSG:4326) for consistency with global coordinates
    clipped_points_crs = clipped_points.to_crs(4326)

    # Return the clipped and reprojected points
    return clipped_points_crs

def assign_centroid_identifiers(clusters, points_for_solar_data):
    """
    This function calculates the centroids of each cluster, reprojects them to WGS 84 (EPSG:4326),
    and assigns a nearest-neighbor identifier from a given set of points (e.g., points for solar data).
    The identifiers from the `points_for_solar_data` dataset are matched to the centroids of the 
    clusters using a k-d tree for efficient nearest-neighbor search.

    Parameters:
    clusters (GeoDataFrame): The input GeoDataFrame containing polygonal clusters.
    points_for_solar_data (GeoDataFrame): GeoDataFrame containing points used for assigning identifiers 
                                          (e.g., solar data centroids) with an 'id' column.

    Returns:
    GeoDataFrame: The original `clusters` GeoDataFrame with an additional 'identifier' column, 
                  which contains the IDs of the nearest points from `points_for_solar_data`.
    """
    
    # Copy clusters and calculate their centroids
    clusters_centroids = clusters.copy()
    clusters_centroids["geometry"] = clusters_centroids.centroid
    clusters_centroids = clusters_centroids.to_crs(4326)  # Reproject to WGS 84 (EPSG:4326)
    
    # Convert points for solar data into an array of coordinates (set A)
    set_a_coords = np.array(list(points_for_solar_data.geometry.apply(lambda geom: (geom.x, geom.y))))
    ckdtree = cKDTree(set_a_coords)  # Create a k-d tree for fast nearest-neighbor lookup
    
    # Convert cluster centroids into an array of coordinates (set B)
    set_b_coords = np.array(list(clusters_centroids.geometry.apply(lambda geom: (geom.x, geom.y))))
    
    # Find the nearest point in set A for each point in set B
    distances, indices = ckdtree.query(set_b_coords)
    
    # Assign the nearest 'id' from points_for_solar_data to each cluster centroid
    clusters_centroids['identifier'] = points_for_solar_data.iloc[indices]['id'].values
    clusters_centroids["identifier"] = clusters_centroids['identifier'].astype(int)
    
    # Merge the identifier back into the original clusters
    clusters_subset = clusters_centroids[['id', 'identifier']]
    clusters = clusters.merge(clusters_subset, how='left', left_on='id', right_on='id')

    return clusters

def getting_solar_data_in_bulk_renewable_ninja(clusters_centroids, workspace_pv, token):
    start_time = time.time()
    for index, row in clusters_centroids.iterrows():
        longitude, latitude = row['geometry'].x, row['geometry'].y
        get_pv_data(latitude, longitude, token, workspace_pv)
    
    end_time = time.time()
    
    # List all elements in the directory
    elements = os.listdir(workspace_pv)
    
    # Count the number of elements
    element_count = len(elements)
    
    total_time = (end_time-start_time)/60
    #
    print("Retrieved {} locations for hourly solar data and temperature readings from renewable.ninja in {} minutes".format(element_count, round(total_time, 1)))


def getting_pv_data_bulk(admin0, clusters, target_crs, workspace_pv, token):
    """
    Processes the administrative boundary, clusters, and solar data to extract and assign solar data
    in bulk. This includes extracting centroids, assigning identifiers, and fetching solar data.

    Parameters:
    admin0 (GeoDataFrame): The administrative boundary geometry.
    clusters (GeoDataFrame): The input GeoDataFrame containing polygonal clusters.
    target_crs (str): The target coordinate reference system for projection.
    workspace_pv (str): The workspace path for solar data.

    Returns:
    clusters (GeoDataFrame): The input GeoDataFrame containing polygonal clusters and identifiers
    """
    
    # Extract centroids for solar data
    points_for_solar_data = centroids_to_extract_solar_data(admin0, target_crs)
    
    # Assign identifiers to cluster centroids
    clusters = assign_centroid_identifiers(clusters, points_for_solar_data)
    
    # Fetch solar data in bulk
    getting_solar_data_in_bulk_renewable_ninja(points_for_solar_data, workspace_pv, token)

    return clusters, points_for_solar_data
         

def estimating_annual_demand_by_bundles(cluster_polygons, households, reasonablecons=1, smeshare=0.08, bundleCshare=0.08, bundleBshare=0.36):
    # reasonablecons: Connectivity rate of structures - for now this value should not be changed 
    # smeshare: Share of SME customers as % of total potential connections
    # bundleCshare: Share of Residential Bundle C customers as % of total potential connections
    # bundleBshare: Share of Residential Bundle B customers as % of total potential connections

    # Define bins and labels
    bins = [0, 20, 50, 150, float('inf')]
    labels = ['<20 m2', '20-50 m2', '50-150 m2', '>150 m2']
    
    # Calculate the statistics without adding a new column
    area_brackets = pd.cut(households['area_in_meters'], bins=bins, labels=labels, right=False)
    bracket_stats = area_brackets.value_counts().sort_index()
    
    # Create a new copy to avoid modifying the original slice
    cluster_polygons_copy = cluster_polygons.copy()
    
    # Add the total sum as value per column
    for label in labels:
        cluster_polygons_copy[label] = bracket_stats.get(label, 0)

    # Perform calculations on the copy
    cluster_polygons_copy.loc[:, "Connections_All"] = np.ceil(cluster_polygons_copy["<20 m2"] + 
                                   cluster_polygons_copy["20-50 m2"] + 
                                   cluster_polygons_copy["50-150 m2"] + 
                                   cluster_polygons_copy[">150 m2"]).astype(int)
    
    cluster_polygons_copy.loc[:, "Potential_Con"] = np.ceil(cluster_polygons_copy["Connections_All"] * reasonablecons)
    
    def assignPUoE(a, b, c, d, e):
        puoe = int(round(-2.913362564 + (-0.00415764*a) + (0.008440603*b) + (0.016433547*c) + (0.722331464*d) + (-0.040674586*e), 0))
        if puoe <= 0:
            return 0
        else:
            return puoe

    cluster_polygons_copy.loc[:, "PUoE"] = cluster_polygons_copy.apply(lambda row: assignPUoE(row["<20 m2"],
                                                                                         row["20-50 m2"],
                                                                                         row["50-150 m2"],
                                                                                         row[">150 m2"],
                                                                                         0), axis=1)
    
    cluster_polygons_copy.loc[:, "SME"] = np.ceil((cluster_polygons_copy["Potential_Con"] - cluster_polygons_copy["PUoE"]) * smeshare)
    
    cluster_polygons_copy.loc[:, "ResC"] = np.ceil((cluster_polygons_copy["Potential_Con"] - cluster_polygons_copy["PUoE"]) * bundleCshare)
    
    cluster_polygons_copy.loc[:, "ResB"] = np.ceil((cluster_polygons_copy["Potential_Con"] - cluster_polygons_copy["PUoE"]) * bundleBshare)
    
    cluster_polygons_copy.loc[:, "ResA"] = (cluster_polygons_copy["Potential_Con"] - cluster_polygons_copy["PUoE"] - 
                                            cluster_polygons_copy["SME"] - cluster_polygons_copy["ResC"] - 
                                            cluster_polygons_copy["ResB"])
    
    # Calculate load curve and annual demand
    load_curve = calc_load_curve(cluster_polygons_copy["ResA"].iloc[0],
                                     cluster_polygons_copy["ResB"].iloc[0],
                                     cluster_polygons_copy["ResC"].iloc[0],
                                     cluster_polygons_copy["SME"].iloc[0],
                                     cluster_polygons_copy["PUoE"].iloc[0])
    
    annual_demand = sum(load_curve)
    
    # Add total demand to the copy
    cluster_polygons_copy["Total demand [kWh/Year]"] = annual_demand

    return cluster_polygons_copy, annual_demand, load_curve


def optimizer_de(diesel_price,
                 hourly_ghi,
                 hourly_temp,
                 load_curve,
                 diesel_cost,  # diesel generator capital cost, USD/kW rated power
                 discount_rate,
                 n_chg,  # charge efficiency of battery
                 n_dis,  # discharge efficiency of battery
                 battery_cost,  # battery capital cost, USD/kWh of storage capacity
                 pv_cost,  # PV panel capital cost, USD/kW peak power
                 charge_controller,  # PV charge controller cost, USD/kW peak power, set to 0 if already included in pv_cost
                 pv_inverter,  # PV inverter cost, USD/kW peak power, set to 0 if already included in pv_cost
                 pv_life,  # PV panel expected lifetime, years
                 diesel_life,  # diesel generator expected lifetime, years
                 pv_om,  # annual OM cost of PV panels
                 diesel_om,  # annual OM cost of diesel generator
                 battery_inverter_cost,
                 battery_inverter_life,
                 dod_max,  # maximum depth of discharge of battery
                 inv_eff,  # inverter efficiency
                 lpsp_max,  # maximum loss of load allowed over the year, in share of kWh
                 diesel_limit,
                 full_life_cycles,
                 start_year,
                 end_year):
    
    demand = load_curve.sum()

    # Define the solution space for the optimization
    battery_bounds = [0, 5 * demand / 365]
    pv_bounds = [0, 5 * max(load_curve)]
    diesel_bounds = [0.5, max(load_curve)]
    
    min_bounds = np.array([pv_bounds[0], battery_bounds[0], diesel_bounds[0]])
    max_bounds = np.array([pv_bounds[1], battery_bounds[1], diesel_bounds[1]])
    bounds = Bounds(min_bounds, max_bounds)

    # Create a series of the hour numbers (0-24) for one year
    hour_numbers = np.empty(8760)
    for i in range(365):
        for j in range(24):
            hour_numbers[i * 24 + j] = j

    # Initialize a list to store additional values
    additional_values_storage = []

    def opt_func(X):
        nonlocal additional_values_storage  # Ensure it references the list in the parent function
        results = find_least_cost_option(X, hourly_temp, hourly_ghi, hour_numbers,
                                         load_curve, inv_eff, n_dis, n_chg, dod_max,
                                         diesel_price, end_year, start_year, pv_cost, charge_controller, pv_inverter, pv_om,
                                         diesel_cost, diesel_om, battery_inverter_life, battery_inverter_cost, diesel_life, pv_life,
                                         battery_cost, discount_rate, lpsp_max, diesel_limit, full_life_cycles)
        
        lcoe = results[0]
        additional_values_storage = results[1:]  # Capture the additional values
                                       
        return lcoe

    minimizer_kwargs = {"method": "BFGS"}
    pv_init = sum(pv_bounds) / 2
    battery_init = sum(battery_bounds) / 2
    diesel_init = sum(diesel_bounds) / 2
    x0 = [pv_init, battery_init, diesel_init]
    
    # Perform optimization using differential evolution
    result = differential_evolution(opt_func, bounds, popsize=15, init='latinhypercube')  # init='halton' on newer env

    best_solution = result.x
    lcoe = opt_func(best_solution)  # This call will also update additional_values_storage
    
    # Return the best solution, LCOE, and additional values
    return {
        "best_solution": best_solution,
        "lcoe": lcoe,
        "additional_values": additional_values_storage
    }


def plotting_jpeg_new(country, name_community, id, No_Households, gpd_mg, service_gdf, secondary_gdf, trunks_gdf, poles, clip_footprints):

    gpd_mg = gpd_mg.to_crs(4326)
    service_gdf = service_gdf.to_crs(4326)
    secondary_gdf = secondary_gdf.to_crs(4326)
    trunks_gdf = trunks_gdf.to_crs(4326)
    poles = poles.to_crs(4326)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each GeoDataFrame
    gpd_mg.plot(ax=ax, edgecolor='black', alpha=0.2)
    service_gdf.plot(ax=ax, edgecolor='black', alpha=0.1)
    secondary_gdf.plot(ax=ax, edgecolor='blue', alpha=0.2)
    trunks_gdf.plot(ax=ax, edgecolor='orange', alpha=0.2)
    poles.plot(ax=ax, color='green', markersize=1)
    
    # Set plot limits based on the total bounds of the main grid
    xmin, ymin, xmax, ymax = gpd_mg.total_bounds
    ax.set_xlim(xmin - 0.001, xmax + 0.001)
    ax.set_ylim(ymin - 0.001, ymax + 0.001)
    
    # Add title and other text
    ax.set_title(f"{country} ({name_community})", size=8)
    ax.text(0.5, -0.1, f"id={id} and No. Households={No_Households}", fontsize=10, ha='center', va='bottom', transform=ax.transAxes)
    
    # Customize tick parameters
    ax.tick_params(axis='x', labelcolor='black', labelsize=6)
    ax.tick_params(axis='y', labelcolor='black', labelsize=6)
    
    # Custom legend for each plotted feature
    legend_handles = [
        mpatches.Patch(color='black', alpha=0.2, label='Cluster'),  # Patch for cluster
        Line2D([0], [0], color='black', lw=2, label='Service Drops'),  # Line for service drops
        Line2D([0], [0], color='blue', lw=2, label='Secondary Lines'),  # Line for secondary lines
        Line2D([0], [0], color='orange', lw=2, label='Trunk Lines'),  # Line for trunk lines
        Line2D([0], [0], color='green', marker='o', lw=0, label='Poles')  # Marker for poles
    ]
    
    # Add custom legend to the plot
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    
    return fig