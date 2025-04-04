#!/usr/bin/env python3
import sys
import math
import heapq
import numpy as np
from collections import deque

#student details
STUDENT_ID = 'a1889102'
DEGREE = 'UG'

#Function to parse through the input map and get the necessary info
def load_map_data(filepath):
    with open(filepath, 'r') as file:
        
        grid_rows, grid_cols = map(int, file.readline().strip().split()) #grid size
        start_point = tuple(map(int, file.readline().strip().split())) #get the start position
        goal_point = tuple(map(int, file.readline().strip().split()))    #get the end position
        grid_map = []
        
        for _ in range(grid_rows):
            grid_map.append(file.readline().rstrip('\n').split())
    return grid_rows, grid_cols, start_point, goal_point, grid_map

#Function to calc the cost to move from one point to another
def movement_cost(grid_map, row1, col1, row2, col2):
    elevation_1 = int(grid_map[row1-1][col1-1])  # elevation of first point
    elevation_2 = int(grid_map[row2-1][col2-1])  #elevation of second point
    return 1 + max(0, elevation_2 - elevation_1)

#function for the manhattan heuristic for a* algo
def manhattan_distance(node, destination):
    (row1, col1), (row2, col2) = node, destination
    return abs(row2 - row1) + abs(col2 - col1)

#function for the euclidean heuristic for the a* algo
def euclidean_distance(node, destination):
    (row1, col1), (row2, col2) = node, destination
    return math.sqrt((row2 - row1)**2 + (col2 - col1)**2)

#function to reconstruct the path from the start to the goal using the parent cell
def backtrack_path(parents, start_point, goal_point):
    if goal_point not in parents: return None  #no path is found since there is no goal
    path_sequence = []
    current = goal_point
    while current is not None:
        path_sequence.append(current)
        current = parents[current] #traverse from the goal to the start
    return list(reversed(path_sequence)) #reverse to get the correct order

#function to get the valid neighbor cells that are n,e,s,w from the current node
def get_neighbors(row, col, grid_rows, grid_cols, grid_map):
    candidates = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
    valid_neighbors = []
    for r, c in candidates:
        if 1 <= r <= grid_rows and 1 <= c <= grid_cols:
            if grid_map[r-1][c-1] != 'X':  #avoid cells with 'X'
                valid_neighbors.append((r, c))
    return valid_neighbors

#breadth-first search function
def bfs_search(grid_rows, grid_cols, grid_map, start_point, goal_point, visit_tracker, first_seen, last_seen):
    queue = deque()
    queue.append(start_point)
    parent_nodes = {start_point: None}
    visit_counter = 0
    row, col = start_point
    visit_counter += 1
    visit_tracker[row-1][col-1] += 1
    if first_seen[row-1][col-1] == 0: first_seen[row-1][col-1] = visit_counter
    last_seen[row-1][col-1] = visit_counter
    while queue:
        current = queue.popleft()
        if current == goal_point: return True, parent_nodes
        cr, cc = current
        for neighbor in get_neighbors(cr, cc, grid_rows, grid_cols, grid_map):
            r, c = neighbor
            visit_counter += 1
            visit_tracker[r-1][c-1] += 1
            if first_seen[r-1][c-1] == 0: first_seen[r-1][c-1] = visit_counter
            last_seen[r-1][c-1] = visit_counter
            if neighbor not in parent_nodes:
                parent_nodes[neighbor] = current
                queue.append(neighbor)
    return False, parent_nodes        #no goal found

# Uniform Cost Search
def ucs_search(num_rows, num_cols, grid, start_pos, goal_pos, visit_tracker, first_visit_log, last_visit_log):
    # Set variables, queue, visited
    priority_queue = []
    tie_resolver = 0
    heapq.heappush(priority_queue, (0, tie_resolver, start_pos))
    path_parent = {start_pos: None}
    min_cost = {start_pos: 0}
    visit_order = 0
    row, col = start_pos
    visit_order += 1
    visit_tracker[row-1][col-1] += 1
    if first_visit_log[row-1][col-1] == 0: first_visit_log[row-1][col-1] = visit_order
    last_visit_log[row-1][col-1] = visit_order      
    while priority_queue:  # Iterate through min heap
        current_cost, _, current_pos = heapq.heappop(priority_queue)
        if current_pos == goal_pos: return True, path_parent  # Found goal, otherwise keep iterating through heap
        curr_row, curr_col = current_pos
        for neighbor in neighbors(curr_row, curr_col, num_rows, num_cols, grid):
            step_cost = cost_function(grid, curr_row, curr_col, neighbor[0], neighbor[1])
            total_cost = current_cost + step_cost
            row, col = neighbor
            visit_order += 1
            visit_tracker[row-1][col-1] += 1
            if first_visit_log[row-1][col-1] == 0: first_visit_log[row-1][col-1] = visit_order
            last_visit_log[row-1][col-1] = visit_order
            if neighbor not in min_cost or total_cost < min_cost[neighbor]:
                min_cost[neighbor] = total_cost
                path_parent[neighbor] = current_pos
                tie_resolver += 1
                heapq.heappush(priority_queue, (total_cost, tie_resolver, neighbor))
    return False, path_parent  # Goal not found

# A* Algorithm
def astar_search(num_rows, num_cols, grid, start_pos, goal_pos, heuristic_fn, visit_tracker, first_visit_log, last_visit_log):
    # Set up variables
    priority_queue = []
    tie_resolver = 0
    heuristic_value = heuristic_fn(start_pos, goal_pos)
    heapq.heappush(priority_queue, (heuristic_value, tie_resolver, 0, start_pos))
    path_parent = {start_pos: None}
    min_cost = {start_pos: 0}
    visit_order = 0
    row, col = start_pos
    visit_order += 1
    visit_tracker[row-1][col-1] += 1
    if first_visit_log[row-1][col-1] == 0: first_visit_log[row-1][col-1] = visit_order
    last_visit_log[row-1][col-1] = visit_order
    while priority_queue:  # Iterate through heap
        f_value, _, g_value, current_pos = heapq.heappop(priority_queue)
        if current_pos == goal_pos: return True, path_parent  # Goal found, otherwise keep iterating through heap
        curr_row, curr_col = current_pos
        for neighbor in neighbors(curr_row, curr_col, num_rows, num_cols, grid):
            step_cost = cost_function(grid, curr_row, curr_col, neighbor[0], neighbor[1])
            new_g_value = g_value + step_cost
            row, col = neighbor
            visit_order += 1
            visit_tracker[row-1][col-1] += 1
            if first_visit_log[row-1][col-1] == 0: first_visit_log[row-1][col-1] = visit_order
            last_visit_log[row-1][col-1] = visit_order
            if neighbor not in min_cost or new_g_value < min_cost[neighbor]:
                min_cost[neighbor] = new_g_value
                path_parent[neighbor] = current_pos
                tie_resolver += 1
                new_f_value = new_g_value + heuristic_fn(neighbor, goal_pos)
                heapq.heappush(priority_queue, (new_f_value, tie_resolver, new_g_value, neighbor))
    return False, path_parent  # No goal found

def print_debug_output(grid_rows, grid_cols, grid_map, path_nodes, visit_tracker, first_time_visit, last_time_visit):
    # print path onto grid
    print("Path:")
    if path_nodes is None:
        print("null")
    else:
        path_set = set(path_nodes)  # convert the path into a set so that lookup is O(1)
        for row_idx in range(grid_rows):
            row_display = []
            for col_idx in range(grid_cols):
                if grid_map[row_idx][col_idx] == 'X':
                    row_display.append('X')  # Keep obstacles as 'X'
                else:
                    row_display.append('*' if (row_idx+1, col_idx+1) in path_set else grid_map[row_idx][col_idx])
            print(" ".join(row_display))
    
    # print the vists of each cell
    print("# Visits:")
    for row_idx in range(grid_rows):
        row_display = []
        for col_idx in range(grid_cols):
            if grid_map[row_idx][col_idx] == 'X':
                row_display.append('X')
            else:
                row_display.append(str(visit_tracker[row_idx][col_idx]) if visit_tracker[row_idx][col_idx] != 0 else '.')
        print(" ".join(row_display))
    
    # print the first time visit for each cell
    print("First Visit:")
    for row_idx in range(grid_rows):
        row_display = []
        for col_idx in range(grid_cols):
            if grid_map[row_idx][col_idx] == 'X':
                row_display.append('X')
            else:
                row_display.append(str(first_time_visit[row_idx][col_idx]) if first_time_visit[row_idx][col_idx] != 0 else '.')
        print(" ".join(row_display))
    
    # print last visit for each cell
    print("Last Visit:")
    for row_idx in range(grid_rows):
        row_display = []
        for col_idx in range(grid_cols):
            if grid_map[row_idx][col_idx] == 'X':
                row_display.append('X')
            else:
                row_display.append(str(last_time_visit[row_idx][col_idx]) if last_time_visit[row_idx][col_idx] != 0 else '.')
        print(" ".join(row_display))

def print_release_output(grid_rows, grid_cols, grid_map, path_nodes):
    # Print the final path only
    if path_nodes is None:
        print("null")
    else:
        path_set = set(path_nodes)  # Convert into set for O(1) lookup
        for row_idx in range(grid_rows):
            row_display = []
            for col_idx in range(grid_cols):
                if grid_map[row_idx][col_idx] == 'X':
                    row_display.append('X')  # Keep obstacles as 'X'
                else:
                    row_display.append('*' if (row_idx+1, col_idx+1) in path_set else grid_map[row_idx][col_idx])
            print(" ".join(row_display))

def main():
    if len(sys.argv) < 4:
        print("Usage: python pathfinder.py [mode] [mapfile] [algorithm] [heuristic]")
        sys.exit(1)
    mode = sys.argv[1]
    mapfile = sys.argv[2]
    algorithm = sys.argv[3]
    if algorithm == 'astar':
        if len(sys.argv) < 5:
            print("For astar, please provide a heuristic: euclidean or manhattan.")
            sys.exit(1)
        heuristic_name = sys.argv[4]
        if heuristic_name == 'euclidean':
            heuristic_fn = euclidean_distance
        elif heuristic_name == 'manhattan':
            heuristic_fn = manhattan_distance
        else:
            print("Unknown heuristic. Use 'euclidean' or 'manhattan'.")
            sys.exit(1)
    else:
        heuristic_fn = None
    rows, cols, start, goal, M = load_map_data(mapfile)
    visit_count = np.zeros((rows, cols), dtype=int)
    first_visit = np.zeros((rows, cols), dtype=int)
    last_visit = np.zeros((rows, cols), dtype=int)
    if algorithm == 'bfs':
        found, parent = bfs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit)
    elif algorithm == 'ucs':
        found, parent = ucs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit)
    elif algorithm == 'astar':
        found, parent = astar_search(rows, cols, M, start, goal, heuristic_fn, visit_count, first_visit, last_visit)
    else:
        print("Unknown algorithm. Use 'bfs', 'ucs', or 'astar'.")
        sys.exit(1)
    path_positions = backtrack_path(parent, start, goal) if found else None
    if mode == 'debug':
        print_debug_output(rows, cols, M, path_positions, visit_count, first_visit, last_visit)
    else:
        print_release_output(rows, cols, M, path_positions)

if __name__ == "__main__":
    main()
