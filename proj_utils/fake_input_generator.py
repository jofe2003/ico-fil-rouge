from random import uniform
import math

def _generate_random_points(num_points, center = (0,0), min_x = 0, max_x = 100, min_y = 0, max_y = 100):
    points = []

    for i in range(num_points):
        points.append((uniform(min_x,max_x)+center[0],
                       uniform(min_y,max_x)+center[1]))

    return points

def _generate_complete_graph(points):
    # Function to calculate distance between two points
    def distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    # Create an empty distance matrix
    adjacency_matrix = [[0 for _ in range(len(points))] for _ in range(len(points))]

    # Fill the distance matrix
    for i in range(len(points)):
        for j in range(i, len(points)):  # Loop only from i to avoid duplicates
            dist = distance(points[i], points[j])
            adjacency_matrix[i][j] = dist
            adjacency_matrix[j][i] = dist
    return adjacency_matrix

def generator(num_points):
    points = _generate_random_points(num_points)
    demand_list = [(0,0)]+[(1,1) for _ in range(num_points-1)]
    adjacency_matrix = _generate_complete_graph(points)

    return adjacency_matrix, demand_list, points