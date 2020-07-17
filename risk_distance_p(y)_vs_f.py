################################################################################

#This contains code for risk-aware travelling salesman problem with submodular
#risk optimization and distance as submodular function

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv

file = open('risk_distance_p(y)_vs_f.csv', 'w')
file2 = open('risk_distance_H_vs_alpha.csv', 'w')
file3 = open('risk_distance_H_vs_alpha_tau_good.csv', 'w')

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
            else:
                i-=1

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

def submodular_fun(e, f, tau, Hf):
    global G
    mu = ((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)/(len(G.tour)+1)**2
    sigma =  500*((len(G.tour)+1)**2)/sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
    expectationfUe = 0
    fUe = 0
    for i in range(100):
        sample = -100
        while True:
            sample = np.random.normal(mu, sigma)
            if mu/10<sample<10*mu:
                break
        t = f + sample
        if t-tau <0:
            expectationfUe += 0
        else:
            expectationfUe += (t-tau)
        fUe += t
    expectationfUe /= 100
    fUe /= 100
    HfUe = tau + expectationfUe/(1-G.alpha)
    f_marginal = HfUe - Hf
    return f_marginal, HfUe, fUe

G = graph(1500, 0.0)

def main():
    print("Number of nodes")
    n = 6 #int(input())
    for i in range(n):
        for j in range(i+1,n):
            G.edges.append([i, j])
    o=0
    i=0
    G.alpha = 0.0
    G.points = []
    G.rand_vert_init(n)
    print(G.points)
    f_min = 0
    for i in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        G.alpha = i
        H_min = 100000
        tour_best = []
        tau_min = 0
        for tau1 in range(0, G.tau, 1):
            print(tau1, G.alpha)
            subtour = [0]*n
            tau = float(tau1)
            G.tour = []
            edges = list(G.edges)
            Hf = tau
            f = 0
            f_marginal = 0
            while edges!=[] and len(G.tour)<=n:
                p = -1
                fm = []
                for e in range(len(edges)):
                    f_marginal, HfUe, fUe = submodular_fun(edges[e], f, tau, Hf)
                    fm.append([f_marginal, HfUe, fUe, e])
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
            print(f)
            if Hf<H_min:
                tau_min = tau
                H_min = Hf
                f_min = f
                tour_best = list(G.tour)
        G.all_tour.append(tour_best)
        print(tour_best)
        file2.write('%f;' % float(H_min))
        file3.write('%f;' % float(tau_min))
        # file.write('%f;' % float(f_min))


    for i in range(len(G.all_tour)):
        G.tour = G.all_tour[i]
        print("T", G.tour)
        temp = 0
        for k in range(100):
            f = 0
            for j in range(len(G.tour)):
                e = G.tour[j]
                mu = ((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)/(j+1)**2
                sigma =  500*((j+1)**2)/sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
                sample = -100
                while True:
                    sample = np.random.normal(mu, sigma)
                    if mu/10<sample<10*mu:
                        break
                f += sample
            temp+=f
            file.write('%f;' % float(f))
        temp = temp/100
        print(temp)
        file.write('\n')

if __name__ == '__main__':
    main()
