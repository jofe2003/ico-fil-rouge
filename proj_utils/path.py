import matplotlib.pyplot as plt

def split_paths(paths):
    path_list= []
    path_id = -1
    for i in paths:
        if(i == 0):
            if(path_id != -1):
                path_list[path_id].append(i)    
            path_list.append([])
            path_id+=1
        path_list[path_id].append(i)
        


    return path_list[:-1]

def print_path(points, sol):
    x = list(map(lambda x: points[x][0], sol))
    y = list(map(lambda x: points[x][1], sol))

    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.35)

    ax.scatter(x,y, marker='x', color = 'red', )

    path_graphs = []
    for path in split_paths(sol):
        x = list(map(lambda x: points[x][0], path))
        y = list(map(lambda x: points[x][1], path))
        path_graph, = ax.plot(x,y)
        path_graphs.append(path_graph)