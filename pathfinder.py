#!/usr/bin/env python3
import sys
import math
import heapq
import numpy as np
from collections import deque

STUDENT_ID = 'a1889102'
DEGREE = 'UG'

def parse_map_file(filepath):
    with open(filepath, 'r') as f:
        rows, cols = map(int, f.readline().strip().split())
        start = tuple(map(int, f.readline().strip().split()))
        goal = tuple(map(int, f.readline().strip().split()))
        M = []
        for _ in range(rows):
            M.append(f.readline().rstrip('\n').split())
    return rows, cols, start, goal, M

def cost_function(M, r1, c1, r2, c2):
    elev1 = int(M[r1-1][c1-1])
    elev2 = int(M[r2-1][c2-1])
    return 1 + max(0, elev2 - elev1)

def manhattan_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return abs(r2 - r1) + abs(c2 - c1)

def euclidean_heuristic(current, goal):
    (r1, c1), (r2, c2) = current, goal
    return math.sqrt((r2 - r1)**2 + (c2 - c1)**2)

def reconstruct_path(parent, start, goal):
    if goal not in parent: return None
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = parent[curr]
    return list(reversed(path))

def neighbors(r, c, rows, cols, M):
    cand = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    valid = []
    for rr, cc in cand:
        if 1 <= rr <= rows and 1 <= cc <= cols:
            if M[rr-1][cc-1] != 'X':
                valid.append((rr, cc))
    return valid

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
        if current == goal: return True, parent
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
    return False, parent

def ucs_search(rows, cols, M, start, goal, visit_count, first_visit, last_visit):
    pq = []
    tiebreaker = 0
    heapq.heappush(pq, (0, tiebreaker, start))
    parent = {start: None}
    cost_so_far = {start: 0}
    counter = 0
    r, c = start
    counter += 1
    visit_count[r-1][c-1] += 1
    if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
    last_visit[r-1][c-1] = counter
    while pq:
        curr_cost, _, current = heapq.heappop(pq)
        if current == goal: return True, parent
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_cost = curr_cost + step
            r, c = nbr
            counter += 1
            visit_count[r-1][c-1] += 1
            if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
            last_visit[r-1][c-1] = counter
            if nbr not in cost_so_far or new_cost < cost_so_far[nbr]:
                cost_so_far[nbr] = new_cost
                parent[nbr] = current
                tiebreaker += 1
                heapq.heappush(pq, (new_cost, tiebreaker, nbr))
    return False, parent

def astar_search(rows, cols, M, start, goal, heuristic, visit_count, first_visit, last_visit):
    pq = []
    tiebreaker = 0
    h = heuristic(start, goal)
    heapq.heappush(pq, (h, tiebreaker, 0, start))
    parent = {start: None}
    cost_so_far = {start: 0}
    counter = 0
    r, c = start
    counter += 1
    visit_count[r-1][c-1] += 1
    if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
    last_visit[r-1][c-1] = counter
    while pq:
        f, _, g, current = heapq.heappop(pq)
        if current == goal: return True, parent
        cr, cc = current
        for nbr in neighbors(cr, cc, rows, cols, M):
            step = cost_function(M, cr, cc, nbr[0], nbr[1])
            new_g = g + step
            r, c = nbr
            counter += 1
            visit_count[r-1][c-1] += 1
            if first_visit[r-1][c-1] == 0: first_visit[r-1][c-1] = counter
            last_visit[r-1][c-1] = counter
            if nbr not in cost_so_far or new_g < cost_so_far[nbr]:
                cost_so_far[nbr] = new_g
                parent[nbr] = current
                tiebreaker += 1
                new_f = new_g + heuristic(nbr, goal)
                heapq.heappush(pq, (new_f, tiebreaker, new_g, nbr))
    return False, parent

def print_debug_output(rows, cols, M, path_positions, visit_count, first_visit, last_visit):
    print("path:")
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
    print("#visits:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(visit_count[r][c]) if visit_count[r][c] != 0 else '.')
        print(" ".join(row))
    print("first visit:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(first_visit[r][c]) if first_visit[r][c] != 0 else '.')
        print(" ".join(row))
    print("last visit:")
    for r in range(rows):
        row = []
        for c in range(cols):
            if M[r][c] == 'X':
                row.append('X')
            else:
                row.append(str(last_visit[r][c]) if last_visit[r][c] != 0 else '.')
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
        print("Unknown algorithm. Use 'bfs', 'ucs', or 'astar'.")
        sys.exit(1)
    path_positions = reconstruct_path(parent, start, goal) if found else None
    if mode == 'debug':
        print_debug_output(rows, cols, M, path_positions, visit_count, first_visit, last_visit)
    else:
        print_release_output(rows, cols, M, path_positions)

if __name__ == "__main__":
    main()
