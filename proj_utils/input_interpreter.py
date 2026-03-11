import pandas as pd
import math

def _calculate_customer_loads(route_id, customer_df):
    df_route1 = customer_df.loc[customer_df['ROUTE_ID'] == route_id] #Filtra pelo primeiro ID

    tab_limites = ['CUSTOMER_CODE','TOTAL_WEIGHT_KG','TOTAL_VOLUME_M3','CUSTOMER_DELIVERY_SERVICE_TIME_MIN']

    df_lim = df_route1[tab_limites].copy() #Tabela com os requisitos de peso, volume e tempo gasto na entrega de cada cliente

    lim_peso = df_lim['TOTAL_WEIGHT_KG'].tolist()
    lim_volume = df_lim['TOTAL_VOLUME_M3'].tolist()
    lim_time = df_lim['CUSTOMER_DELIVERY_SERVICE_TIME_MIN'].tolist()

    lista_limites = [(0,0)]+list(zip(lim_peso,lim_volume))
    return lista_limites

def _calculate_dist_adjacency_matrix(route_id, customer_df, deposit_df ):
    df_route = customer_df.loc[customer_df['ROUTE_ID'] == route_id] #Filtra a tabela de clientes pelo primeiro ID
    df_depot = deposit_df.loc[deposit_df['ROUTE_ID'] == route_id] #filtra a tabela de depot pelo ID
    #Começa o tratamento de dados do dataframe

    tab_dist = ['CUSTOMER_LATITUDE','CUSTOMER_LONGITUDE']
    tab_depot = ['DEPOT_LATITUDE','DEPOT_LONGITUDE']

    df_dist = df_route[tab_dist].copy()
    df_depot = df_depot[tab_depot].copy()

    df_dist.columns,df_depot.columns  = ['LATITUDE', 'LONGITUDE'],['LATITUDE', 'LONGITUDE']

    latitude_list = df_dist['LATITUDE'].tolist()
    latitude_list.append(df_depot['LATITUDE'].tolist()[0])

    longitude_list = df_dist['LONGITUDE'].tolist()
    longitude_list.append(df_depot['LONGITUDE'].tolist()[0])

    #Desliza 1 unidade para direita, para que o deposito fique na primeira posição

    latitude_list = latitude_list[-1:] + latitude_list[:-1]
    longitude_list = longitude_list[-1:] + longitude_list[:-1]

    points = list(zip(latitude_list, longitude_list))
    #Termina o tratamento de dados do dataframe e cria um novo
    
    #TODO: CALCULATE DISTANCE IN TIME INSTEAD OF DISTANCE

    new_df = pd.DataFrame({'LATITUDE': latitude_list, 'LONGITUDE': longitude_list})

    distance_matrix = []

    for i in range(len(latitude_list)):
        distance_matrix.append([])
        for j in range(len(longitude_list)):
            distance_matrix[i].append(math.sqrt((longitude_list[i] - longitude_list[j]) ** 2 +
                                        (latitude_list[i] - latitude_list[j]) ** 2))

    return distance_matrix, points

def _calculate_time_adjacency_matrix(route_id, customer_df, deposit_df, vel):
    df_route = customer_df.loc[customer_df['ROUTE_ID'] == route_id] #Filtra a tabela de clientes pelo primeiro ID
    df_depot = deposit_df.loc[deposit_df['ROUTE_ID'] == route_id] #filtra a tabela de depot pelo ID
    #Começa o tratamento de dados do dataframe

    tab_dist = ['CUSTOMER_LATITUDE','CUSTOMER_LONGITUDE']
    tab_depot = ['DEPOT_LATITUDE','DEPOT_LONGITUDE']

    df_dist = df_route[tab_dist].copy()
    df_depot = df_depot[tab_depot].copy()

    df_dist.columns,df_depot.columns  = ['LATITUDE', 'LONGITUDE'],['LATITUDE', 'LONGITUDE']

    latitude_list = df_dist['LATITUDE'].tolist()
    latitude_list.append(df_depot['LATITUDE'].tolist()[0])

    longitude_list = df_dist['LONGITUDE'].tolist()
    longitude_list.append(df_depot['LONGITUDE'].tolist()[0])

    #Desliza 1 unidade para direita, para que o deposito fique na primeira posição

    latitude_list = latitude_list[-1:] + latitude_list[:-1]
    longitude_list = longitude_list[-1:] + longitude_list[:-1]

    #Termina o tratamento de dados do dataframe e cria um novo
    points = list(zip(latitude_list, longitude_list))
    new_df = pd.DataFrame({'LATITUDE': latitude_list, 'LONGITUDE': longitude_list})

    temp_matrix = []

    for i in range(len(latitude_list)):
        temp_matrix.append([])
        for j in range(len(longitude_list)):
            aux = math.sqrt((longitude_list[i] - longitude_list[j]) ** 2 + (latitude_list[i] - latitude_list[j]) ** 2)
            temp_matrix[i].append(aux/vel)

    return temp_matrix, points

def _calculate_delivery_windows(route_id, customer_df):
    df_route1 = customer_df.loc[customer_df['ROUTE_ID'] == route_id]
    tab_delivery_widow = ['CUSTOMER_TIME_WINDOW_FROM_MIN', 'CUSTOMER_TIME_WINDOW_TO_MIN']
    df_delivery_window = df_route1[tab_delivery_widow].copy()

    list_window_start = (df_delivery_window['CUSTOMER_TIME_WINDOW_FROM_MIN']-480).tolist()
    list_window_end = (df_delivery_window['CUSTOMER_TIME_WINDOW_TO_MIN']-480).tolist()
    
    #res = [(0,float('inf'))] +list(zip([0 for _ in range(len(list_window_start))], [float('inf') for _ in range(len(list_window_end))]))
    #TODO: DELETE THIS WHEN THE GRAPH IS IN TIME INSTEAD OF DISTANCE
    res = [(0,float('inf'))] +list(zip([0 for _ in range(len(list_window_start))], list_window_end))

    return res

def format_input(route_id, customer_df, deposit_df):
    customer_loads = _calculate_customer_loads(route_id, customer_df)
    adjacency_matrix,points = _calculate_time_adjacency_matrix(route_id, customer_df, deposit_df, 0.6)
    delivery_window = _calculate_delivery_windows(route_id, customer_df)
    return adjacency_matrix, customer_loads, points