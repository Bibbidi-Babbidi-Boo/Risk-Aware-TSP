################################################################################

#This contains code for risk-aware travelling salesman problem with submodular
#risk optimization and 1/distance as submodular function

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt

file = open('risk_inverse_distance.csv', 'w')

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
    mu, sigma = 0, 100*sqrt(2)/(len(G.points)*sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2))
    expectationfUe = 0
    fUe = 0
    for i in range(100):
        sample = np.random.normal(mu, sigma)
        t = (f+(100*(1+sample)*sqrt(2)/((len(G.tour)+1)*sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2))))
        if tau - t <0:
            expectationfUe += 0
        else:
            expectationfUe += (tau-t)
        fUe += t
    expectationfUe /= 100
    fUe /= 100
    HfUe = tau - expectationfUe/G.alpha
    f_marginal = HfUe - Hf
    return f_marginal, HfUe, fUe

G = graph(600, 0.9)

def main():
    print("Number of nodes")
    n = 10 #int(input())
    # G.rand_vert_init(n)
    for i in range(n):
        for j in range(i+1,n):
            G.edges.append([i, j])
    o=0
    i=0
    # for o in range(20):
    G.alpha = 0.0
    G.rand_vert_init(n)
    for i in range(9):
        G.alpha += 0.1
        G.all_tour_costs = []
        G.all_tau = []
        G.tour = []
        G.all_tour = []
        for tau1 in range(20, G.tau, 1):
            subtour = [0]*n
            tau = float(tau1)
            print(tau)
            G.tour = []
            edges = list(G.edges)
            Hf = tau*(1-1/G.alpha)
            f = 0
            # f_p = 0
            f_marginal = 0
            t = 0
            while edges!=[] and len(G.tour)<=n:
                p = 0
                fm = []
                for e in range(len(edges)):
                    f_marginal, HfUe, fUe = submodular_fun(G.tour, edges[e], f, tau, Hf)
                    fm.append([f_marginal, e, HfUe, fUe])
                fm.sort(reverse=1)
                p = fm[0][1]
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
                        Hf = fm[0][2]
                        f = fm[0][3]
                edges.pop(p)
            if G.tour != []:
                G.all_tour.append(G.tour)
                G.all_tour_costs.append(Hf)
                G.all_tau.append(tau)
            print("HF", Hf)
            # file.write('%f;' % float(f))
            file.write('%f;' % float(Hf))
        # print(G.all_tour_costs)
        # print(max(G.all_tour_costs))
        file.write('\n')
    # print(G.all_tour_costs)
    # print(max(G.all_tour_costs))
    # pos = G.all_tour_costs.index(max(G.all_tour_costs))
    # # print(pos, len(G.all_tour_costs))
    # # pos = np.argmax(G.all_tour_costs, axis=0)
    # f = G.all_tour_costs[pos]
    # G.tour = G.all_tour[pos]
    # G.tau = G.all_tau[pos]
    # print(G.tour, Hf, G.tau)
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
