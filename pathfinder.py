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
    with open(filepath, 'r') as f:

        rows, cols = map(int, f.readline().strip().split()) #grid size
        start = tuple(map(int, f.readline().strip().split())) #get the start position
        goal = tuple(map(int, f.readline().strip().split()))    #get the end position
        M = []
        
        for _ in range(rows):
            M.append(f.readline().rstrip('\n').split())
    return rows, cols, start, goal, M

#Function to calc the cost to move from one point to another
def cost_of_movement(M, r1, c1, r2, c2):
    elevation1 = int(M[r1-1][c1-1])  # elevation of first point
    elevation2 = int(M[r2-1][c2-1])  #elevation of second point
    return 1 + max(0, elevation2 - elevation1)

#function for th emanhattan heuristic for a* algo
def manhattan_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return abs(r2 - r1) + abs(c2 - c1)

#function for the eucilidean hueristic for the a* algo
def euclidean_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return math.sqrt((r2 - r1)**2 + (c2 - c1)**2)

#function to reconstruct the path from the start to the goal using the parent cell
def backtrack_path(parent, start, goal):
    if goal not in parent: return None  #no path is found since there is no goal
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr] #traverse from them goal to the start
    return list(reversed(path)) #therefore u need to reverse the order to get the correct path

#function to get the vaild neighbour cells that are n,e,s,w from the current node
def neighbors(r, c, rows, cols, M):
    directions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)] #adding one or subtracting one from the cell coords
    valid = []
    for rr, cc in directions:
        if 1 <= rr <= rows and 1 <= cc <= cols:
            if M[rr-1][cc-1] != 'X':    #don't go onto cells with 'X'
                valid.append((rr, cc))
    return valid


#breath first search function
def bfs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    #set up all variables, queue, start cells and visited_count
    q = deque()
    q.append(start)
    parent = {start: None}
    count = 0
    r, c = start
    count += 1
    visit_count[r-1][c-1] += 1
    if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = count
    last_visit[r-1][c-1] = count
    while q:                        #iterate through the queue
        current = q.popleft()
        if current == goal: return True, parent     #check for goal, otherwise keep iterating
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            r, c = nbr
            count += 1
            visit_count[r-1][c-1] += 1
            if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = count
            last_visit[r-1][c-1] = count
            if nbr not in parent:
                parent[nbr] = current
                q.append(nbr)
    return False, parent        #no goal found

#union cost search
def ucs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    #set variables, queue, visted
    pq = []
    tie_breaker = 0
    heapq.heappush(pq, (0, tie_breaker, start))
    parent = {start: None}
    cost_so_far = {start: 0}
    count = 0
    r, c = start
    count += 1
    visit_count[r-1][c-1] += 1
    if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = count
    last_visit[r-1][c-1] = count      
    while pq:                   #iterate through minheap
        current_cost, _, current = heapq.heappop(pq)
        if current == goal: return True, parent     #found goal, otherwise keep iterating thorugh heap
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_of_movement(M, cr, cc, nbr[0], nbr[1])
            new_cost = current_cost + step
            r, c = nbr
            count += 1
            visit_count[r-1][c-1] += 1
            if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = count
            last_visit[r-1][c-1] = count
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = current
                tie_breaker += 1
                heapq.heappush(pq, (new_cost, tie_breaker, nbr))
    return False, parent        #goal not found

#a* algo
def astar_search(rows, cols, M, start, goal, heuristic, visit_count, visit_first, visit_last):
    #set up varibales
    pq = []
    tie_breaker = 0
    h = heuristic(start, goal)
    heapq.heappush(pq, (h, tie_breaker, 0, start))
    parent = {start: None}
    cost_till_now = {start: 0}
    count = 0
    r, c = start
    count += 1
    visit_count[r-1][c-1] += 1
    if visit_first[r-1][c-1] == 0: visit_first[r-1][c-1] = count
    visit_last[r-1][c-1] = count
    while pq:       #itetate through heap
        f, _, g, current = heapq.heappop(pq)
        if current == goal: return True, parent         #goal found, otherwise keep iterating through heap
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_of_movement(M, cr, cc, nbr[0], nbr[1])
            new_g = g + step
            r, c = nbr
            count += 1
            visit_count[r-1][c-1] += 1
            if visit_first[r-1][c-1] == 0: visit_first[r-1][c-1] = count
            visit_last[r-1][c-1] = count
            if nbr not in cost_till_now or new_g < cost_till_now[nbr]:
                cost_till_now[nbr] = new_g
                parent[nbr] = current
                tie_breaker += 1
                new_f = new_g + heuristic(nbr, goal)
                heapq.heappush(pq, (new_f, tie_breaker, new_g, nbr))
    return False, parent        #no goal found

def print_debug_output(rows, cols, M, path_pos, visit_num, visit_first, visit_last):
    print("path:")
    if path_pos is None:
        print("null")
    else:
        set_path = set(path_pos)
        for r in range(rows):
            row = []
            for c in range(cols):
                if M[r][c] == 'X':
                    row.append('X')
                else:
                    row.append('*' if (r+1, c+1) in set_path else M[r][c])
            print(" ".join(row))
    print("#visits:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(visit_num[r][c]) if visit_num[r][c] != 0 else '.')
        print(" ".join(row))
    print("first visit:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(visit_first[r][c]) if visit_first[r][c] != 0 else '.')
        print(" ".join(row))
    print("last visit:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(
                    [r][c]) if visit_last[r][c] != 0 else '.')
        print(" ".join(row))

def print_release_output(rows, cols, M, path_positions):
    if path_positions is None:
        print("null")
    else:
        path_set = set(path_positions)
        for r in range(rows):
            row = []
            for c in range(cols):
                if M[r][c] == 'X':
                    row.append('X')
                else:
                    row.append('*' if (r+1, c+1) in path_set else M[r][c])
            print(" ".join(row))

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
            heuristic_fn = euclidean_heuristic
        elif heuristic_name == 'manhattan':
            heuristic_fn = manhattan_heuristic
        else:
            print("Unknown heuristic. Use 'euclidean' or 'manhattan'.")
            sys.exit(1)
    else:
        heuristic_fn = None
    rows, cols, start, goal, M = load_map_data(mapfile)
    visit_count = np.zeros((rows, cols), dtype=int)
    visit_first = np.zeros((rows, cols), dtype=int)
    visit_last = np.zeros((rows, cols), dtype=int)
    if algorithm == 'bfs':
        found, parent = bfs_search(rows, cols, M, start, goal, visit_count, visit_first, visit_last)
    elif algorithm == 'ucs':
        found, parent = ucs_search(rows, cols, M, start, goal, visit_count, visit_first, visit_last)
    elif algorithm == 'astar':
        found, parent = astar_search(rows, cols, M, start, goal, heuristic_fn, visit_count, visit_first, visit_last)
    else:
        print("Unknown algorithm. Use 'bfs', 'ucs', or 'astar'.")
        sys.exit(1)
    path_pos = backtrack_path(parent, start, goal) if found else None
    if mode == 'debug':
        print_debug_output(rows, cols, M, path_pos, visit_count, visit_first, visit_last)
    else:
        print_release_output(rows, cols, M, path_pos)

if __name__ == "__main__":
    main()
