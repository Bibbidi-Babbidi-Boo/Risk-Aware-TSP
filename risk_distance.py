################################################################################

#This contains code for risk-aware travelling salesman problem with submodular
#risk optimization and distance as submodular function

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv

file = open('risk_distance_best_tau.csv', 'w')

class graph():
    def __init__(self, tau, alpha):
        self.tau = tau
        self.alpha = alpha
        self.points = []
        self.edges = []
        self.all_tour_costs = []
        self.all_tau = []
        self.tour = []
        self.all_tour = []

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

def submodular_fun(S, e, f, tau, Hf):
    global G
    mu, sigma = 0, (len(G.points)/(sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)))
    expectationfUe = 0
    fUe = 0
    for i in range(100):
        sample = np.random.normal(mu, sigma)
        t = f + (1+sample)*sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)/((len(G.tour)+1)**2)
        if t-tau <0:
            expectationfUe += 0
        else:
            expectationfUe += (t-tau)
        fUe += t
    expectationfUe /= 100
    fUe /= 100
    HfUe = tau + expectationfUe/(1-G.alpha)
    f_marginal = HfUe - Hf
    # print("EX", f, fUe, Hf, HfUe, f_marginal)
    return f_marginal, HfUe, fUe

G = graph(30, 0.0)

def main():
    print("Number of nodes")
    n = 10 #int(input())
    for i in range(n):
        for j in range(i+1,n):
            G.edges.append([i, j])
    # print(G.points)
    o=0
    i=0
    for o in range(1):
        G.alpha = 0.0
        G.points = []
        G.rand_vert_init(n)
        for i in range(9):
            G.alpha += 0.1
            G.all_tour_costs = []
            G.all_tau = []
            G.tour = []
            G.all_tour = []
            print("Alpha", G.alpha)
            for tau1 in range(0, G.tau*100, 5):
                subtour = [0]*n
                tau = float(tau1/100)
                # print(tau)
                G.tour = []
                edges = list(G.edges)
                Hf = tau
                f = 0
                f_marginal = 0
                while edges!=[] and len(G.tour)<=n:
                    p = -1
                    fm = []
                    for e in range(len(edges)):
                        f_marginal, HfUe, fUe = submodular_fun(G.tour, edges[e], f, tau, Hf)
                        fm.append([f_marginal, HfUe, fUe, e])
                    # print("------------------------")
                    fm.sort()
                    p = fm[0][3]
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
                            Hf = fm[0][1]
                            f = fm[0][2]
                    edges.pop(p)
                if G.tour != []:
                    G.all_tour.append(G.tour)
                    G.all_tour_costs.append(Hf)
                    G.all_tau.append(tau)
                # print("HF", Hf)
                # # file.write('%f;' % float(f))
                # file.write('%f;' % float(Hf))
            min(G.all_tour_costs)
            print("HF", min(G.all_tour_costs))
            file.write('%f;' % float(min(G.all_tour_costs)))
            file.write('\n')
    # pos = G.all_tour_costs.index(min(G.all_tour_costs))
    # f = G.all_tour_costs[pos]
    # G.tour = G.all_tour[pos]
    # G.tau = G.all_tau[pos]
    # print(G.tour, f, G.tau)
    # x = []
    # y = []
    # for e in G.tour:
    #     x = [G.points[e[0]][0], G.points[e[1]][0]]
    #     y = [G.points[e[0]][1], G.points[e[1]][1]]
    #     plt.plot(x, y, color='r')
    # for p in G.points:
    #     plt.plot(p[0], p[1], marker='o', color='b')
    # plt.show()
    # G.tour = []

if __name__ == '__main__':
    main()
