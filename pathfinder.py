#!/usr/bin/env python3
import sys
import math
import heapq
import numpy as np
from collections import deque

# Student details
STUDENT_ID = 'a1889102'
DEGREE = 'UG'

# function to be able to parse through the map
def parse_map_file(filepath):
    with open(filepath, 'r') as f:
        rows, cols = map(int, f.readline().strip().split())  # go through the grid dimensions
        start = tuple(map(int, f.readline().strip().split()))  # check start positon
        goal = tuple(map(int, f.readline().strip().split()))  # check the end position
        M = []  # intialize the map
        for _ in range(rows):
            M.append(f.readline().rstrip('\n').split())  # go through the grid rows
    return rows, cols, start, goal, M

# function to calc the cost of going form 1 cell to the other
def cost_function(M, r1, c1, r2, c2):
    elev1 = int(M[r1-1][c1-1])  # elevation of first
    elev2 = int(M[r2-1][c2-1])  # elevation of second
    return 1 + max(0, elev2 - elev1)  # base 1 penalty

# manhattan heuristic for a* algo
def manhattan_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return abs(r2 - r1) + abs(c2 - c1)

# euclidiean hueristic for a* algo
def euclidean_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return math.sqrt((r2 - r1)**2 + (c2 - c1)**2)

# reconstruct from parent dictionary
def reconstruct_path(parent, start, goal):
    if goal not in parent: return None  # check to see if the goal is reached
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]  # Backtrack from goal to start
    return list(reversed(path))  # reverse the order

# all the negihbouring cells
def neighbors(r, c, rows, cols, M):
    cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]  # N,E,S,W
    valid = []
    for rr, cc in cand:
        if 1 <= rr <= rows and 1 <= cc <= cols:
            if M[rr-1][cc-1] != 'X':  # dont go on X
                valid.append((rr, cc))
    return valid

# bfs
def bfs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    q = deque()
    q.append(start)
    parent = {start: None}
    counter = 0
    r, c = start
    counter += 1
    visit_count[r-1][c-1] += 1
    if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
    last_visit[r-1][c-1] = counter
    
    while q:
        current = q.popleft()
        if current == goal: return True, parent  # if goal is found, otherwise keep iterating through queue
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            r, c = nbr
            counter += 1
            visit_count[r-1][c-1] += 1
            if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
            last_visit[r-1][c-1] = counter
            if nbr not in parent:
                parent[nbr] = current
                q.append(nbr)
    return False, parent  # goal doesnt exist

# ucs
def ucs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    pq = []
    tiebreaker = 0
    heapq.heappush(pq, (0, tiebreaker, start))  # Min heap
    parent = {start: None}
    cost_so_far = {start: 0}
    counter = 0
    
    while pq:
        curr_cost, _, current = heapq.heappop(pq)
        if current == goal: return True, parent #if goal found, otherwise keep iterating through the minheap
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_cost = curr_cost + step
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = current
                tiebreaker += 1
                heapq.heappush(pq, (new_cost, tiebreaker, nbr))
    return False, parent #goal doesnt exist

# A*
def astar_search(rows, cols, M, start, goal, heuristic, visit_count, first_visit, last_visit):
    pq = []
    tiebreaker = 0
    h = heuristic(start, goal)
    heapq.heappush(pq, (h, tiebreaker, 0, start))
    parent = {start: None}
    cost_so_far = {start: 0}
    
    while pq:
        f, _, g, current = heapq.heappop(pq)
        if current == goal: return True, parent #if goal found, otherwise keep iterating through heap
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_g = g + step
            if nbr not in cost_so_far or new_g < cost_so_far[nbr]:
                cost_so_far[nbr] = new_g
                parent[nbr] = current
                tiebreaker += 1
                new_f = new_g + heuristic(nbr, goal)
                heapq.heappush(pq, (new_f, tiebreaker, new_g, nbr))
    return False, parent #goal doesnt exist

# mai fucntion to call all searcing functions
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
        heuristic_fn = manhattan_heuristic if heuristic_name == 'manhattan' else euclidean_heuristic
    else:
        heuristic_fn = None
    
    rows, cols, start, goal, M = parse_map_file(mapfile)
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
        print("Unknown algorithm.")
        sys.exit(1)

    path_positions = reconstruct_path(parent, start, goal) if found else None
    print_release_output(rows, cols, M, path_positions)

if __name__ == "__main__":
    main()
