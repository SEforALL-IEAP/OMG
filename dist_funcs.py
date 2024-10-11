import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, cKDTree
from shapely.geometry import LineString, Point, MultiLineString, Polygon, MultiPoint
from shapely.ops import linemerge, nearest_points, voronoi_diagram, unary_union, split, substring
from shapely import minimum_rotated_rectangle, unary_union
import networkx as nx
import math
import warnings

def convert_multipoint_to_point(geometry):
    # This functions converts a MultiPoint to Point
    if isinstance(geometry, MultiPoint):
        # Take the centroid of the MultiPoint
        return geometry.centroid
    else:
        # Return the original geometry
        return geometry

def polygon_rotation(polygon):
    # This function takes a Polygon as an input, identifies the minimum rectangle that encapsules that polygon,
    # and returns the rotations and side lengths of that rectangle. 
    
    # Get the minimum bounding box of the polygon
    bounding_box = minimum_rotated_rectangle(polygon)

    # Get the coordinates of the bounding box vertices
    bounding_box_coords = np.array(bounding_box.exterior.coords)

    # Calculate the length of each side of the minimum rotated rectangle
    side_lengths = np.linalg.norm(np.diff(bounding_box_coords, axis=0), axis=1)

    # Find the maximum length among these sides
    max_length = np.max(side_lengths)
    min_length = np.min(side_lengths)

    # Calculate the vectors of the bounding box edges
    edges = np.diff(bounding_box_coords, axis=0)

    # Calculate the angle of rotation (in radians)
    angle_radians_w = np.arctan2(edges[1, 1], edges[1, 0])
    angle_radians_l = np.arctan2(edges[2, 1], edges[2, 0])

    x_start = bounding_box_coords[0][0]
    y_start = bounding_box_coords[0][1]

    return angle_radians_w, angle_radians_l, max_length, min_length, x_start, y_start

def ckdnearest(gdA, gdB):
    # This method finds the nearest point of set B for each point in set A 

    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
    gdf = pd.concat(
        [
            gdA.reset_index(drop=True),
            gdB_nearest,
            pd.Series(dist, name='dist')
        ], 
        axis=1)

    return gdf

def voronoi_areas(simple_trunk, polygon, spacing=25, plot=True):
    # This function takes a Polygon and the trunk line within that polygon, 
    # converts the trunk line to a set of points at a specified spacing distance, 
    # then uses voronoi polygons to find which part of the polygon is closest to each segment of the trunk line
    
    points_sorted = {}
    points_all = []
    for id, line in simple_trunk.iterrows():
        distances = np.arange(0, line.geometry.length, spacing)
        points = [line.geometry.interpolate(distance) for distance in distances]
        points_sorted[id] = points
        points_all += points
    mp = MultiPoint(points_all)

    vor_regions = voronoi_diagram(mp, envelope=polygon)

    sorted_regions = {}
    regions = []
    for reg in vor_regions.geoms:
        regions.append(reg)
    
    for key in points_sorted:
        points = points_sorted[key]
        for point in points:
            for reg in regions:
                if reg.contains(point):
                    try:
                        sorted_regions[key].append(reg)
                    except:
                        sorted_regions[key] = [reg]
                    regions.remove(reg)  

    dissolved_regions = []
    fids = []
    
    for key in sorted_regions:
        dissolved = unary_union(sorted_regions[key])
        dissolved = dissolved.intersection(polygon)
        dissolved_regions.append(dissolved)
        fids.append(key)
    
    dissolved_regions_gdf = gpd.GeoDataFrame(geometry=dissolved_regions)
    if plot:
        dissolved_regions_gdf.plot()

    return dissolved_regions

def create_trunk_line(polygon, spacing=50, plot=True):
    
    # Extract the boundary of the polygon
    boundary = polygon.boundary
    
    # Step 3: Generate evenly spaced points along the boundary
    # Get the length of the boundary
    boundary_length = boundary.length
    
    # Generate points along the boundary
    points = []
    distance = 0
    while distance < boundary_length:
        point = boundary.interpolate(distance)
        points.append(point)
        distance += spacing
    
    # Step 4: Convert the points to a GeoDataFrame
    points_gdf = gpd.GeoDataFrame(geometry=points)
    
    # Convert the list of points to a NumPy array
    point_array = np.array([[point.x, point.y] for point in points])
    
    minx, miny, maxx, maxy = polygon.bounds
    
    # Calculate padding as a fraction of the bounding box size (e.g., 10% of the max dimension)
    width = maxx - minx
    height = maxy - miny
    padding = max(width, height) * 0.1  # 10% of the larger dimension
    
    extra_points = np.array([
        [minx - padding, miny - padding],
        [maxx + padding, miny - padding],
        [maxx + padding, maxy + padding],
        [minx - padding, maxy + padding]
    ])
    
    all_points = np.vstack([point_array, extra_points])
    
    vor = Voronoi(all_points, furthest_site=False)
    
    # Create a list to hold the polygons
    voronoi_polygons = []
    
    # Iterate over Voronoi regions and intersect them with the polygon
    for region in vor.regions:
        #if len(region) > 2:
        if not -1 in region and len(region) > 0:
            # Extract the vertices of the region
            vertices = [vor.vertices[i] for i in region]
            # Create a polygon from the vertices
    
            voronoi_polygon = Polygon(vertices)
            # Intersect the polygon with the given polygon
            try:
                intersection = voronoi_polygon.intersection(polygon)
                if intersection.is_empty:
                    continue
                if intersection.geom_type == 'Polygon':
                    voronoi_polygons.append(intersection)
                elif intersection.geom_type == 'MultiPolygon':
                    voronoi_polygons.append(intersection)
            except:  # ToDo
                pass
    
    
    # Create a GeoDataFrame from the Voronoi polygons
    gdf_voronoi = gpd.GeoDataFrame(geometry=voronoi_polygons)
    
    # Create a list to hold the line segments that do not intersect with the exterior of the single polygon
    lines = []
    
    # Buffer polygon boundary by 10 m 
    #boundary_polygon = polygon.geometry.iloc[0].boundary.buffer(distance=10)
    boundary_polygon = polygon.boundary.buffer(distance=10)  
    
    # Iterate over each polygon in the GeoDataFrame and remove lines from the voronoi polygons that intersect with the buffered polygon
    for polygon in gdf_voronoi.geometry:
        # Get the boundary of the voronoi polygon
        polygon_boundary = polygon.boundary
        try: # ToDo MultiPolygon???
            # Split the boundary into line segments
            for i in range(len(polygon_boundary.coords) - 1):
                part_start = polygon_boundary.coords[i]
                part_end = polygon_boundary.coords[i + 1]
                line = LineString([part_start, part_end])
                line_reverse = LineString([part_end, part_start])
                # Check if the segment intersects with the exterior of the single polygon
                if not line.intersects(boundary_polygon):
                    if (line not in lines) & (line_reverse not in lines):
                        lines.append(line)
        except:
            pass
    
    # Create a MultiLineString from the list of lines
    multiline = MultiLineString(lines)
    
    # Create a GeoDataFrame from the MultiLineString
    trunk_lines = gpd.GeoDataFrame(geometry=[multiline])
    if plot:
        # Plot the result
        ax = gdf_voronoi.plot(alpha=0.5, edgecolor='k')
        points_gdf.plot(ax=ax, color='blue', markersize=50, label='Boundary Points')
        trunk_lines.plot(ax=ax, color='red', label='Trunk')
        ax.legend()
        
    return trunk_lines

def simplify_trunk_lines(trunks, length_removal=100, split_distance=500, plot=True):
    # This function simplifies the trunk lines, by removing branches of the trunk line that are shorter than a specified threshold.
    
    verts = []  # Store the vertices

    # Get all the line segments of the MultiLineString
    trunk_linestrings = trunks.explode(index_parts=True)
    trunk_linestrings.reset_index(inplace=True)
    
    # Iterate over all the LineString segments and add vertices to list
    for id, line in trunk_linestrings.iterrows():
        verts.append(Point(line.geometry.coords[0]))
        verts.append(Point(line.geometry.coords[1]))
    
    # Count the occurrence of each vertice
    feature_counts = {}
    for feature in verts:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1
    
    intersections = []

    # If a vertice appear more than twice, it's an intersections. Att that vertice to the intersections list
    for key in feature_counts:
        if feature_counts[key] > 2: 
             intersections.append(key)
    
    int_gdf = gpd.GeoDataFrame(geometry=intersections)

    # Create a MultiPoint of all intersections
    ints = MultiPoint(intersections)

    # Perform linemerge on the trunk MultiLine to combine LineStrings
    merged_lines = linemerge(trunk_linestrings.geometry.tolist())

    # Then split at the intersections
    split_lines = split(merged_lines, ints)

    # Keep only the split lines that start and end at an intersection point, or that are "longer" than a specified length 
    spl = []
    for line in split_lines.geoms:
        if (Point(line.coords[0]) in intersections) & (Point(line.coords[-1]) in intersections):
            spl.append(line)
        elif line.length > length_removal:
            spl.append(line)

    try:
        spl = list(linemerge(spl).geoms)
    except AttributeError:
        pass
    spl_2 = []
    
    for line in spl:
        if line.length > split_distance * 1.5:
            segments_no = math.floor(line.length / split_distance)
            distance = line.length / (segments_no + 1)
            split_points = []
            for i in range(segments_no + 1):
                subs = substring(line, i * distance, (i+1) * distance)
                spl_2.append(subs)
        else:
            spl_2.append(line)

    final_lines = gpd.GeoDataFrame(geometry=spl_2)
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        trunk_linestrings.plot(ax=ax)
        int_gdf.plot(ax=ax, color='red')
        final_lines.plot(ax=ax, color='black')

    return spl_2, intersections

def assign_households(candidate_poles, households):
    # This function assigns households to the nearest candidate pole, then returns the selected poles and the required service drops.
    
    candidates_gdf = gpd.GeoDataFrame(geometry=candidate_poles)
    assigned_poles = []  # This contains all the poles to which at least one building was assigned
    service_drops = []  # This contains all service drops
    
    gdf_coords = list(zip(candidates_gdf.geometry.x, candidates_gdf.geometry.y))
    tree = cKDTree(gdf_coords)

    try:
        for idx, household in households.iterrows():
            # Find the nearest pole to the household
            distance, index = tree.query((household.geometry.x, household.geometry.y))
            nearest_pole = candidates_gdf.iloc[index]
                
            # Add the nearest pole to the assigned_poles GeoDataFrame
            if candidates_gdf.iloc[index].geometry not in assigned_poles:
                assigned_poles.append(candidates_gdf.iloc[index].geometry)
            
            service_drops.append(LineString([candidates_gdf.iloc[index].geometry, household.geometry]))
    except:
        pass

    return assigned_poles, service_drops

def lv_lines_mst(poles, trunk_poles, assigned_poles, angle_radians_w, angle_radians_l, weight, spacing=50, plot=True):
    # This function creates an MST that connects all of the selected poles (i.e. poles to which at least one building is connected=
    # to the primary trunk lines. The weights of the MST is assigned so that it favors in order 1) connection of two lines on the primary trunk,
    # 2) connection of poles to a pole on the trunk line if they are nearby, 3) connection of two poles that are aligned on the x- or y-axis of 
    # the minimum bounding rectangle to favor orthogonal lines from the trunk and 4) any other connections of two poles.
    
    lv_lines = []
    
    # Create a graph from the GeoDataFrame of points
    G = nx.Graph()

    all_poles = ckdnearest(gpd.GeoDataFrame(geometry=poles), gpd.GeoDataFrame(geometry=trunk_poles))
    
    # Add nodes (points) to the graph
    for idx, point in all_poles.iterrows():
        G.add_node(idx, pos=(point.geometry.x, point.geometry.y))

    # Calculate the Euclidean distance between points and add edges to the graph
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                p_u = all_poles.geometry.iloc[u]
                p_v = all_poles.geometry.iloc[v]
                distance = p_u.distance(p_v)

                if distance < 3 * spacing:
                    ang = np.arctan2(p_u.y - p_v.y, p_u.x - p_v.x)
                    orth = False
                    if (abs(round(ang, 4)) == abs(round(angle_radians_w, 4))) or (abs(round(ang, 4)) == abs(round(angle_radians_l, 4))):
                        orth=True
    
                    distance = p_u.distance(p_v)
                    if (p_u in trunk_poles) and (p_v in trunk_poles):
                        G.add_edge(u, v, weight=0.0001*distance)
                    elif ((p_u in trunk_poles) & (distance / spacing < 1.6) & orth) or ((p_v in trunk_poles) & (distance / spacing < 1.6) & orth):
                    #elif ((all_poles.geometry.iloc[u] in trunk_poles)) or ((all_poles.geometry.iloc[v] in trunk_poles)):
                        G.add_edge(u, v, weight=weight*distance)
                    elif orth:
                        weight_factor_u = (all_poles.dist.iloc[u] / spacing) / 10 
                        weight_factor_v = (all_poles.dist.iloc[v] / spacing) / 10
                        G.add_edge(u, v, weight=distance * (1 + weight_factor_u + weight_factor_v))
                    else:
                        weight_factor_u = (all_poles.dist.iloc[u] / spacing) / 10
                        weight_factor_v = (all_poles.dist.iloc[v] / spacing) / 10
                        G.add_edge(u, v, weight=distance * (1 + weight_factor_u + weight_factor_v) * 2)
                else:
                    G.add_edge(u, v, weight=100 * distance)

    #target_points = trunk_poles
    target_points = []
    for point in assigned_poles:
        target_points.append(point)

    target_points = target_points + trunk_poles

    targets = []
    for idx, geom in all_poles.iterrows():
        if geom.iloc[0] in target_points:
            targets.append(idx)

    # Create a copy of the graph
    G_copy = G.copy()
    
    # Iterate over the points and remove those that are not necessary to connect all selected points
    for node in G.nodes():
        if node not in targets:
            # Temporarily remove the node from the graph
            G_copy.remove_node(node)
            # Check if all selected points are still connected
            if not nx.is_connected(G_copy.subgraph(targets)):
                # If not connected, the removed node is necessary, so add it back
                G_copy.add_node(node)
        else:
            # Add edges to connect selected points to each other
            for u in targets:
                if u != node:
                    G_copy.add_edge(u, node, weight=G[u][node]['weight'])
    
    # Find the Minimum Spanning Tree (MST) of the modified graph
    mst_edges = list(nx.minimum_spanning_tree(G_copy).edges())
    mst_poles = []
    
    if plot:
        # Plot the original points
        fig, ax = plt.subplots(figsize=(10, 10))
        
        all_poles.plot(ax=ax, color='blue')
        
        # Plot edges of the MST
        for edge in mst_edges:
            u, v = edge
            x1, y1 = G.nodes[u]['pos']
            x2, y2 = G.nodes[v]['pos']
            ax.plot([x1, x2], [y1, y2], color='red')
    
            if Point(x1,y1) not in mst_poles:
                mst_poles.append(Point(x1,y1))
            if Point(x2,y2) not in mst_poles:
                mst_poles.append(Point(x2,y2))    
            
            if (Point(x1,y1) in trunk_poles) & (Point(x2, y2) in trunk_poles):
                pass
            else:
                lv_lines.append(LineString([Point(x1,y1), Point(x2, y2)]))
        
        plt.show()
    
    else:
        for edge in mst_edges:
            u, v = edge
            x1, y1 = G.nodes[u]['pos']
            x2, y2 = G.nodes[v]['pos']
            if Point(x1,y1) not in mst_poles:
                mst_poles.append(Point(x1,y1))
            if Point(x2,y2) not in mst_poles:
                mst_poles.append(Point(x2,y2))    
            
            if (Point(x1,y1) in trunk_poles) & (Point(x2, y2) in trunk_poles):
                pass
            else:
                lv_lines.append(LineString([Point(x1,y1), Point(x2, y2)]))

    return lv_lines, mst_poles

def create_candidate_poles(polygon, trunk, distance, buffer=50, plot=True):
    # This function creates a grid of candidate poles within the selected Polygon (which may be a sub-area of a larger polygon).
    # The poles are aligned along the x- and y-axis of the minimum bounding rectangle that contains the polygon.
    # Poles that are within a specified buffer distance from the trunk lines are omitted, as buildings within this distance are assumed to connect  
    # directly to a pole on the trunk line.
    
    mesh_lines_2 = []
    mesh_lines_3 = []
    poles = []
    candidate_poles = {}
    road_points_all = {}

    # Create a buffer around the trunk line to ensure no poles are within that buffer. Households nearby connect directly to the trunk poles
    buffered_trunk = trunk.buffer(buffer)    

    # Find the orientation of the polygon
    angle_radians_w, angle_radians_l, polygon_length, polygon_width, start_x, start_y = polygon_rotation(polygon)
    bounding_box = minimum_rotated_rectangle(polygon)

    # the number of candidate poles to be spaced out length-wise and width-wise
    length_points = polygon_length / distance
    width_points = polygon_width / distance

    no_p = max(length_points, width_points)

    for w in range(math.ceil(no_p)+1):
        for l in range(math.ceil(no_p)+1):
            x = start_x + w * distance * np.cos(angle_radians_w) - l * distance * np.cos(angle_radians_l)
            y = start_y  + w * distance * np.sin(angle_radians_w) - l * distance * np.sin(angle_radians_l)
            
            point = Point(x, y)  # This is a candidate pole (if not within the buffer distance from the trunk)
            if polygon.contains(point):
                if point.within(buffered_trunk):
                    pass
                else:
                    poles.append(point)

            # This creates lines connecting points in the mesh as a grid
            x_next_1 = start_x + (w + 1) * distance * np.cos(angle_radians_w) - l * distance * np.cos(angle_radians_l)
            y_next_1 = start_y  + (w + 1) * distance * np.sin(angle_radians_w) - l * distance * np.sin(angle_radians_l)

            x_next_2 = start_x + w * distance * np.cos(angle_radians_w) - (l + 1) * distance * np.cos(angle_radians_l)
            y_next_2 = start_y  + w * distance * np.sin(angle_radians_w) - (l + 1) * distance * np.sin(angle_radians_l)
            
            point_2 = Point(x_next_1, y_next_1)  # This is the next pole in one direction
            point_3 = Point(x_next_2, y_next_2)  # This is the next pole in the other direction

            mesh_lines_2.append(LineString([point, point_2]))  
            mesh_lines_3.append(LineString([point, point_3]))

    mesh_lines_gdf_2 = gpd.GeoDataFrame(geometry=mesh_lines_2)
    mesh_lines_gdf_3 = gpd.GeoDataFrame(geometry=mesh_lines_3)

    intersection_points_2 = []
    
    # Iterate over the linestrings in the mesh grid to find where they intersect with the trunk, and adds poles at those locations
    for idx, linestring in mesh_lines_gdf_2.iterrows():
        # Check if the linestring intersects with the multilinestring
        if linestring['geometry'].intersects(unary_union(trunk)):
            # If there is an intersection, get the intersection points
            intersection = linestring['geometry'].intersection(unary_union(trunk))
            # If the intersection is a point, store it in the GeoDataFrame
            if intersection.geom_type == 'Point':
                intersection_points_2.append(intersection)
            elif intersection.geom_type == 'MultiPoint':
                print('MultiPoint')
            else:
                print(intersection.geom_type)
                #for p in intersection.geoms:
                #    print(p)
                #pass  # ToDo in case there are more than one intersection

    intersection_points_3 = []
    
    # Iterate over the linestrings in the mesh grid to find where they intersect with the trunk, and adds poles at those locations
    for idx, linestring in mesh_lines_gdf_3.iterrows():
        # Check if the linestring intersects with the multilinestring
        if linestring['geometry'].intersects(unary_union(trunk)):
            # If there is an intersection, get the intersection points
            intersection = linestring['geometry'].intersection(unary_union(trunk))
            # If the intersection is a point, store it in the GeoDataFrame
            if intersection.geom_type == 'Point':
                intersection_points_3.append(intersection)
            elif intersection.geom_type == 'MultiPoint':
                print('MultiPoint')
            else:
                print(intersection.geom_type)
                #for p in intersection.geoms:
                #    print(p)
                #pass  # ToDo in case there are more than one intersection
    

    if len(intersection_points_2) > len(intersection_points_3):
        intersection_points = intersection_points_2
    else:
        intersection_points = intersection_points_3

    #intersection_points = intersection_points_2 + intersection_points_3

    intersection_points.append(Point(trunk.coords[0]))
    intersection_points.append(Point(trunk.coords[-1]))
    
    all_candidates = poles + intersection_points
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))

        all_candidates_gdf = gpd.GeoDataFrame(geometry=all_candidates)
        bounding_gdf = gpd.GeoSeries(bounding_box)
        bounding_gdf.plot(ax=ax, color='green')
        polygon_gdf = gpd.GeoSeries(polygon)
        polygon_gdf.plot(ax=ax, color='gray')
        trunk_gdf = gpd.GeoSeries(trunk)
        trunk_gdf.plot(ax=ax, color='red')
        all_candidates_gdf.plot(ax=ax)

        plt.show()

    return all_candidates, intersection_points, poles, angle_radians_w, angle_radians_l

def creating_grid(trunk_lines, voronois, community, households_centroids, target_crs, pole_dist=50, buffer=25):
    trunk_p = []
    assigned_p = []
    lv_l = []
    service_l = []
    mst_p = []
    all_p = []
    long_services = 0
    
    multi_trunks = []
    multi_trunks_len = []
    
    multi_secondary = []
    multi_secondary_len = []
    
    multi_service = []
    multi_service_len = []
    
    multi_poles = []
    multi_all_poles = []

    if len(trunk_lines) == len(voronois):
        for id in range(len(trunk_lines)):
            # Create candidate poles along the trunk line within the Voronoi cell
            all_poles, trunk_poles, poles, angle_radians_w, angle_radians_l = create_candidate_poles(
                voronois[id], trunk_lines[id], pole_dist, buffer=buffer, plot=False)
        
            # Append the newly created trunk poles and all poles to the respective lists
            trunk_p += trunk_poles
            all_p += all_poles
                
            # Clip household centroids to the current Voronoi cell
            polygon_households = households_centroids.clip(voronois[id]).copy()  # Ensure it's a copy
        
            # Convert MultiPoint geometries to Point if necessary
            polygon_households.loc[:, 'geometry'] = polygon_households['geometry'].apply(convert_multipoint_to_point)

            # Assign households to the nearest poles and calculate service drops
            assigned_poles, service_drops = assign_households(all_poles, polygon_households)
        
            # Append the assigned poles and service drops to their respective lists
            assigned_p += all_poles
            service_l += service_drops
        
            # Count service lines longer than 70 units
            for s in service_drops:
                if s.length > 70:
                    long_services += 1
        
            # Weighting factor for the minimum spanning tree (MST)
            weight = 0.5
        
            # Generate low-voltage lines (secondary lines) using MST of poles
            lv_lines, mst_poles = lv_lines_mst(
                all_poles, trunk_poles, assigned_poles, angle_radians_w, angle_radians_l, weight, plot=False)

            lv_gdf=gpd.GeoSeries(lv_lines)
            lv_gdf = lv_gdf.buffer(10)
            all_pole_gdf=gpd.GeoSeries(all_poles)
            intersecting_points = []
            # Loop over each point in points_gs
            for point in all_pole_gdf:
                # Check if this point intersects with any line in lines_gs
                if lv_gdf.intersects(point).any():
                    intersecting_points.append(point)
            for p in trunk_poles:
                if p not in intersecting_points:
                    intersecting_points.append(p)
        
            # Append the MST poles and low-voltage lines to their respective lists
            lv_l += lv_lines
            mst_p += intersecting_points # mst_poles
    
    else:
        for vor_regions in voronois:
            for line in trunk_lines:
                intersection = vor_regions.intersection(line)
                if not intersection.is_empty and isinstance(intersection, LineString):
                    # Calculate the percentage of overlap
                    intersection_length = intersection.length
                    total_line_length = line.length
                    overlap_percentage = (intersection_length / total_line_length) * 100
        
                    if overlap_percentage >70:
                        #print(f"The line overlaps {overlap_percentage:.2f}% of its length with the polygon.")
                        
                        all_poles, trunk_poles, poles, angle_radians_w, angle_radians_l =\
                        create_candidate_poles(vor_regions, line, pole_dist, buffer=25, plot=False)
        
                        trunk_p += trunk_poles
                        all_p += all_poles
                            
                        polygon_households = households_centroids.clip(vor_regions)
                    
                        # Ensure households are not MultiPoint
                        polygon_households.loc[:, 'geometry'] = polygon_households['geometry'].apply(convert_multipoint_to_point)
                    
                        assigned_poles, service_drops = assign_households(all_poles, polygon_households)
                    
                        assigned_p += all_poles
                        service_l += service_drops
                    
                        for s in service_drops:
                            if s.length > 70:
                                long_services += 1
                    
                        weight = 0.5  # Weighting factor for the MST
                    
                        lv_lines, mst_poles = lv_lines_mst(all_poles, trunk_poles, assigned_poles, angle_radians_w, angle_radians_l, weight, plot=False)
                    
                        lv_gdf=gpd.GeoSeries(lv_lines)
                        lv_gdf = lv_gdf.buffer(10)
                        all_pole_gdf=gpd.GeoSeries(all_poles)
                        intersecting_points = []
                        # Loop over each point in points_gs
                        for point in all_pole_gdf:
                            # Check if this point intersects with any line in lines_gs
                            if lv_gdf.intersects(point).any():
                                intersecting_points.append(point)
                        for p in trunk_poles:
                            if p not in intersecting_points:
                                intersecting_points.append(p)
                    
                        # Append the MST poles and low-voltage lines to their respective lists
                        lv_l += lv_lines
                        mst_p += intersecting_points # mst_poles
    
    # Combine all generated poles into a MultiPoint geometry
    multi_all_poles.append(MultiPoint(all_p))

    # Combine the trunk lines into a MultiLineString geometry and calculate their lengths
    multi_trunks.append(MultiLineString(trunk_lines))
    multi_trunks_len.append(MultiLineString(trunk_lines).length)
    
    # Combine the secondary lines into a MultiLineString geometry and calculate their lengths
    multi_secondary.append(MultiLineString(lv_l))
    multi_secondary_len.append(MultiLineString(lv_l).length)
    
    # Combine the service lines into a MultiLineString geometry and calculate their lengths
    multi_service.append(MultiLineString(service_l))
    multi_service_len.append(MultiLineString(service_l).length)
    
    # Combine the MST poles into a MultiPoint geometry
    multi_poles.append(MultiPoint(mst_p))

    # Create GeoDataFrame for trunk lines
    trunks_gdf = gpd.GeoDataFrame({
        'Length': multi_trunks_len,
        'Type': "Trunk Line",
        'geometry': multi_trunks,
        'id':  community
    })
    trunks_gdf.set_crs(target_crs, inplace=True)  # Set the coordinate reference system (CRS)

    # Create GeoDataFrame for secondary lines
    secondary_gdf = gpd.GeoDataFrame({
        'Length': multi_secondary_len,
        'Type': "Secondary Line",
        'geometry': multi_secondary,
        'id': community
    })
    secondary_gdf.set_crs(target_crs, inplace=True)

    # Create GeoDataFrame for service lines
    service_gdf = gpd.GeoDataFrame({
        'Length': multi_service_len,
        'Type': "Service Line",
        'geometry': multi_service,
        'id': community
    })
    service_gdf.set_crs(target_crs, inplace=True)

    # Create GeoDataFrame for poles
    poles_gdf = gpd.GeoDataFrame({
        'geometry': multi_poles,
        'No. Poles': [len(p.geoms) for p in multi_poles],
        'id': community
    })
    poles_gdf.set_crs(target_crs, inplace=True)

    # Combine all line geometries (trunk, secondary, service) into a single GeoDataFrame
    total_grid_gdf = gpd.GeoDataFrame(pd.concat([trunks_gdf, secondary_gdf, service_gdf], ignore_index=True))

    # Return the combined grid GeoDataFrame, poles GeoDataFrame, and individual line GeoDataFrames
    return total_grid_gdf, poles_gdf, service_gdf, secondary_gdf, trunks_gdf



def plotting_jpeg(country, name_community, id, No_Households, gpd_mg, service_gdf, secondary_gdf, trunks_gdf, poles, clip_footprints):

    gpd_mg=gpd_mg.to_crs(4326)
    service_gdf=service_gdf.to_crs(4326)
    secondary_gdf=secondary_gdf.to_crs(4326)
    trunks_gdf=trunks_gdf.to_crs(4326)
    poles=poles.to_crs(4326)
    #clip_footprints=clip_footprints.to_crs(4326)

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each GeoDataFrame with a label
    gpd_mg.plot(ax=ax, edgecolor='black', alpha=0.2, label='Main Grid')
    #clip_footprints .plot(ax=ax, edgecolor='red', alpha=0.4, label='Building footprints')
    service_gdf.plot(ax=ax, edgecolor='black', alpha=0.1, label='Service drops')
    secondary_gdf.plot(ax=ax, edgecolor='blue', alpha=0.2, label='Secondary lines')
    trunks_gdf.plot(ax=ax, edgecolor='orange', alpha=0.2, label='Trunk lines')
    poles.plot(ax=ax, color='green', label='Poles', markersize=1)
    
    # Set the aspect ratio
    #ax.set_aspect('equal', 'box')
    
    xmin, ymin, xmax, ymax = gpd_mg.total_bounds
    
    ax.set_xlim(xmin-0.001, xmax+0.001)
    ax.set_ylim(ymin-0.001, ymax+0.001)
    
    # Add a title
    txt = ax.set_title(country +'({})'.format(name_community), size=8)
    ax.text(0.5, -0.1, "id={} and No. Households ={}".format(id, No_Households), fontsize=10, ha='center', va='bottom', transform=ax.transAxes)
    #ax.text(0.5, -0.2, "No. Households={}".format(row.NUMPOINTS), fontsize=10, ha='center', va='bottom', transform=ax.transAxes)
       
    
    # Customize tick parameters
    ax.tick_params(axis='x', labelcolor='black', labelsize=6)
    ax.tick_params(axis='y', labelcolor='black', labelsize=6)
    
    # Add the legend
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


    return fig
