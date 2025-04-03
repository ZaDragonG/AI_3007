#!/usr/bin/env python3

import sys
import math
import heapq
import numpy as np
from collections import deque

# Student details
STUDENT_ID = 'a1889102'
DEGREE = 'UG'

# Function to parse the map file
def parse_map_file(filepath):
    with open(filepath, 'r') as f:
        rows, cols = map(int, f.readline().strip().split())  # Grid dimensions
        start = tuple(map(int, f.readline().strip().split()))  # Start position
        goal = tuple(map(int, f.readline().strip().split()))  # Goal position
        M = []  # Initialize the map
        
        for _ in range(rows):
            M.append(f.readline().rstrip('\n').split())  # Read grid rows
    
    return rows, cols, start, goal, M

# Function to calculate the cost of moving between two cells
def cost_function(M, r1, c1, r2, c2):
    elev1 = int(M[r1 - 1][c1 - 1])  # Elevation of first cell
    elev2 = int(M[r2 - 1][c2 - 1])  # Elevation of second cell
    return 1 + max(0, elev2 - elev1)  # Base cost + elevation penalty

# Manhattan heuristic for A* algorithm
def manhattan_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return abs(r2 - r1) + abs(c2 - c1)

# Euclidean heuristic for A* algorithm
def euclidean_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)

# Reconstruct the path from the parent dictionary
def reconstruct_path(parent, start, goal):
    if goal not in parent:
        return None  # Goal not reachable
    
    path = []
    curr = goal
    
    while curr is not None:
        path.append(curr)
        curr = parent[curr]  # Backtrack from goal to start
    
    return list(reversed(path))  # Reverse to get correct order

# Get all valid neighboring cells
def neighbors(r, c, rows, cols, M):
    candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]  # N, S, W, E
    valid_neighbors = []
    
    for rr, cc in candidates:
        if 1 <= rr <= rows and 1 <= cc <= cols and M[rr - 1][cc - 1] != 'X':
            valid_neighbors.append((rr, cc))
    
    return valid_neighbors

# Breadth-First Search (BFS) algorithm
def bfs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    q = deque()
    q.append(start)
    parent = {start: None}
    counter = 1
    
    r, c = start
    visit_count[r - 1][c - 1] += 1
    first_visit[r - 1][c - 1] = counter
    last_visit[r - 1][c - 1] = counter
    
    while q:
        current = q.popleft()
        
        if current == goal:
            return True, parent  # Goal found
        
        cr, cc = current
        
        for nbr in neighbors(cr, cc, rows, cols, M):
            if nbr not in parent:
                parent[nbr] = current
                q.append(nbr)
    
    return False, parent  # Goal not found

# Uniform Cost Search (UCS) algorithm
def ucs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    pq = []
    heapq.heappush(pq, (0, 0, start))  # Min heap
    parent = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        curr_cost, _, current = heapq.heappop(pq)
        
        if current == goal:
            return True, parent  # Goal found
        
        cr, cc = current
        
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_cost = curr_cost + step
            
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = current
                heapq.heappush(pq, (new_cost, _, nbr))
    
    return False, parent  # Goal not found

# A* Search algorithm
def astar_search(rows, cols, M, start, goal, heuristic, visit_count, first_visit, last_visit):
    pq = []
    h = heuristic(start, goal)
    heapq.heappush(pq, (h, 0, 0, start))
    parent = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        f, _, g, current = heapq.heappop(pq)
        
        if current == goal:
            return True, parent  # Goal found
        
        cr, cc = current
        
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_g = g + step
            
            if nbr not in cost_so_far or new_g < cost_so_far[nbr]:
                cost_so_far[nbr] = new_g
                parent[nbr] = current
                new_f = new_g + heuristic(nbr, goal)
                heapq.heappush(pq, (new_f, _, new_g, nbr))
    
    return False, parent  # Goal not found

# Main function to execute the search algorithm
def main():
    if len(sys.argv) < 4:
        print("Usage: python pathfinder.py [mode] [mapfile] [algorithm] [heuristic]")
        sys.exit(1)
    
    mode = sys.argv[1]
    mapfile = sys.argv[2]
    algorithm = sys.argv[3]
    
    if algorithm == 'astar':
        if len(sys.argv) < 5:
            print("For A*, please provide a heuristic: euclidean or manhattan.")
            sys.exit(1)
        
        heuristic_fn = manhattan_heuristic if sys.argv[4] == 'manhattan' else euclidean_heuristic
    else:
        heuristic_fn = None
    
    rows, cols, start, goal, M = parse_map_file(mapfile)
    visit_count = np.zeros((rows, cols), dtype=int)
    first_visit = np.zeros((rows, cols), dtype=int)
    last_visit = np.zeros((rows, cols), dtype=int)
    
    search_algorithms = {'bfs': bfs_search, 'ucs': ucs_search, 'astar': astar_search}
    
    if algorithm in search_algorithms:
        found, parent = search_algorithms[algorithm](rows, cols, M, start, goal, heuristic_fn, visit_count, first_visit, last_visit)
    else:
        print("Unknown algorithm.")
        sys.exit(1)
    
    path_positions = reconstruct_path(parent, start, goal) if found else None
    print(path_positions)

if __name__ == "__main__":
    main()
