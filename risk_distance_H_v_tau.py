################################################################################

#This contains code for risk-aware travelling salesman problem with submodular
#risk optimization and distance as submodular function

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv

file = open('risk_distance_H_v_tau.csv', 'w')

class graph():
    def __init__(self, tau, alpha):
        self.tau = tau
        self.alpha = alpha
        self.points = []
        self.edges = []
        self.all_tour_costs = []
        self.tour = []
        self.all_tour = []
        self.all_tau = []

    def rand_vert_init(self, n):
        for i in range(n):
            x = random.randint(0,100)
            y = random.randint(0,100)
            p = [x,y]
            if p not in self.points:
                self.points.append(p)

    def DFS(self):
        visited = [False]*len(self.points)
        path = list(self.tour)
        edge = [-100, -100]
        start = 0
        ret = True
        while ret == True and path != []:
            i = path[0]
            edge = i
            start = i[0]
            vertex1 = i[0]
            vertex2 = i[1]
            ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
        return ret

    def recursive_next(self, edge, path, visited, start, vertex1, vertex2):
        visited[vertex1] = True
        flag = 0
        path.remove(edge)
        for j in range(len(path)):
            if path[j][0] == vertex2 or path[j][1] == vertex2:
                flag+=1
                if path[j][0] == vertex2:
                    vertex1 = path[j][0]
                    vertex2 = path[j][1]
                else:
                    vertex1 = path[j][1]
                    vertex2 = path[j][0]
                edge = path[j]
                break
        if vertex2 == start:
            return False
        if flag == 0:
            return True
        ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
        return ret

def find_good_path(e, f):
    global G
    mu = sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
    sigma =  10000/sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
    fUe = 0
    sample = -10
    for i in range(100):
        sample = -100
        while True:
            sample = np.random.normal(mu, sigma)
            if 0<sample:
                break
        fUe += sample
    fUe /= 100
    f_marginal = fUe - f
    return f_marginal, fUe

def cost_calc():
    global G
    expect = []
    for j in range(100000):
        f = 0
        for i in range(len(G.tour)):
            e = G.tour[i]
            mu = sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
            sigma =  10000/sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
            sample = -100
            while True:
                sample = np.random.normal(mu, sigma)
                if 0<sample:
                    break
            f += sample
        expect.append(f)
    return expect

G = graph(1000, 0.0)

def main():
    n = 10
    G.rand_vert_init(n)
    for i in range(n):
        for j in range(i+1,n):
            G.edges.append([i, j])
    subtour = [0]*n
    edges = list(G.edges)
    f = 0
    f_marginal = 0
    while edges!=[] and len(G.tour)<=n:
        p = -1
        fm = []
        for e in range(len(edges)):
            f_marginal, fUe = find_good_path(edges[e], f)
            fm.append([f_marginal, fUe, e])
        fm.sort()
        p = fm[0][2]
        if subtour[edges[p][0]]<2 and subtour[edges[p][1]]<2:
            G.tour.append(edges[p])
            subtour[edges[p][0]]+=1
            subtour[edges[p][1]]+=1
            ret = G.DFS()
            if ret == False and len(G.tour)<n:
                G.tour.pop(len(G.tour)-1)
                subtour[edges[p][0]]-=1
                subtour[edges[p][1]]-=1
            else:
                f_marginal = fm[0][0]
                f = fm[0][1]
        edges.pop(p)

    expect = cost_calc()
    for o in range(9):
        G.alpha += 0.1
        for tau1 in range(0, int(max(expect)), 1):
            tau = float(tau1)
            H = 0
            for i in range(len(expect)):
                if expect[i]>tau:
                    H+=expect[i]-tau
                else:
                    H+=0
            H /= len(expect)
            H = tau + H/(1-G.alpha)
            print("H", H, tau)
            file.write('%f;' % float(H))
        file.write('\n')

if __name__ == '__main__':
    main()
