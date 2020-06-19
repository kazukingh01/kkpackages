from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import pandas as pd

def cal_rho(lon_a,lat_a,lon_b,lat_b):
    ra=6378.140  # equatorial radius (km)
    rb=6356.755  # polar radius (km)
    F=(ra-rb)/ra # flattening of the earth
    rad_lat_a=np.radians(lat_a)
    rad_lon_a=np.radians(lon_a)
    rad_lat_b=np.radians(lat_b)
    rad_lon_b=np.radians(lon_b)
    pa=np.arctan(rb/ra*np.tan(rad_lat_a))
    pb=np.arctan(rb/ra*np.tan(rad_lat_b))
    xx=np.arccos(np.sin(pa)*np.sin(pb)+np.cos(pa)*np.cos(pb)*np.cos(rad_lon_a-rad_lon_b))
    c1=(np.sin(xx)-xx)*(np.sin(pa)+np.sin(pb))**2/np.cos(xx/2)**2
    c2=(np.sin(xx)+xx)*(np.sin(pa)-np.sin(pb))**2/np.sin(xx/2)**2
    dr=F/8*(c1-c2)
    rho=ra*(xx+dr)
    return float(rho)

def create_data_model(world_data):
    """Stores the data for the problem."""
    import pandas as pd
    import numpy as np


    df = pd.read_csv(world_data)
    capital_only = df[df['iscapital'] == 1]
    country_pos = {}
    for index, row in capital_only.iterrows():
        country_pos.update({row['name_en']: (row['lat'], row['lon'])})

    distance_dict = {}
    for key in country_pos:
        distance_array = []
        for key2 in country_pos:
            pos1 = country_pos[key]
            pos2 = country_pos[key2]
            distance = cal_rho(country_pos[key][1], country_pos[key][0], country_pos[key2][1], country_pos[key2][0])
            distance_array.append(distance)
        distance_dict.update({key: distance_array})
    name_country = []
    for key in distance_dict:
        name_country.append(key)
        
        
    data = {}
    data['distance_matrix'] = []
    for key in distance_dict:
        data['distance_matrix'].append(distance_dict[key])
    data['num_vehicles'] = 1
    data['starts'] = [5]
    data['ends'] = [5]
    return data, country_pos, name_country


def draw_on_folium(fol_map, step, lat_now, lon_now, lat_prev=None, lon_prev=None):
    import folium
    folium.Marker(location=[lat_now, lon_now], popup=folium.Popup(html=str(step), max_width="50%", show=True)).add_to(fol_map)
    if lat_prev is not None and lon_prev is not None:
        folium.PolyLine(locations=[[lat_prev, lon_prev], [lat_now, lon_now]], weight=1).add_to(fol_map)
    return fol_map
    

def print_solution(manager, routing, solution, country_dict, name_country):
    """Prints solution on console."""
    print('Objective: {} kilometers'.format(solution.ObjectiveValue()))
    
    import folium
    world_map = folium.Map() # 世界地図の作成
    
    index = routing.Start(0)
    plan_output = 'Route for vehicle 0:\n'
    route_distance = 0
    
    step = 0
    lat_s = None
    lon_s=None
    while not routing.IsEnd(index):
        plan_output += ' {} ->'.format(name_country[manager.IndexToNode(index)])
        
        lat_e, lon_e = country_dict[name_country[manager.IndexToNode(index)]][0], country_dict[name_country[manager.IndexToNode(index)]][1]
        if lat_s is None and lon_s is None:
            draw_on_folium(world_map, step, lat_e, lon_e)
        else:
            draw_on_folium(world_map, step, lat_e, lon_e, lat_s, lon_s)
        lat_s, lon_s = country_dict[name_country[manager.IndexToNode(index)]][0], country_dict[name_country[manager.IndexToNode(index)]][1]
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
        step += 1
    plan_output += ' {}\n'.format(manager.IndexToNode(index))
    lat_e, lon_e = country_dict[name_country[manager.IndexToNode(index)]][0], country_dict[name_country[manager.IndexToNode(index)]][1]
    if 'lat_s' in locals() and 'lon_s' in locals():
        draw_on_folium(world_map, step, lat_e, lon_e)
    else:
        draw_on_folium(world_map, step, lat_e, lon_e, lat_s, lon_s)
    print(plan_output)
    
    
    world_map.save('google-OR.html')

    plan_output += 'Route distance: {}kilometers\n'.format(route_distance)


if __name__ == "__main__":
    """Entry point of the program."""
    # Instantiate the data problem.
    world_data = "./now_setting.csv"
    data, country_dict, name_country = create_data_model(world_data)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['starts'], data['ends'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution, country_dict, name_country)
