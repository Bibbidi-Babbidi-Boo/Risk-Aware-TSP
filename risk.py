################################################################################

#This contains code for risk-aware travelling salesman problem with submodular
#risk optimization

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt

class graph():
    def __init__(self, tau):
        self.tau = tau
        self.points = []
        self.edges = []
        self.all_tour_costs = []
        self.tour = []
        self.all_tour = []
        self.recursion_stack = []

    def rand_vert_init(self, n):
        for i in range(n):
            x = random.randint(0,1000)
            y = random.randint(0,1000)
            p = [x,y]
            if p not in self.points:
                self.points.append(p)

    def DFS(self):
        visited = [False]*len(self.points)
        path = list(self.tour)
        edge = [-100, -100]
        start = 0
        ret = True
        # print("p", path)
        for i in range(len(path)):
            if ret == True and path!=[]:
                # print(i, path)
                edge = path[i]
                start = path[i][0]
                vertex1 = path[i][0]
                vertex2 = path[i][1]
                ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
                return ret
            else:
                break
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
                # print("n", edge)
                break
        if vertex2 == start:
            return False
        if flag == 0:
            return True
        ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
        return ret

def submodular_fun(S, e, f):
    global G
    mu, sigma = 0, 20
    expectation = 0
    for i in range(100):
        sample = np.random.normal(mu, sigma)
        # print("Len", len(G.points))
        expectation += (1+sample)/sqrt((G.points[e[0]][0]-G.points[e[1]][0])**2 + (G.points[e[0]][1]-G.points[e[1]][1])**2)
    expectation /= 100
    f += expectation
    f_marginal = f+expectation-f
    return f_marginal, f

G = graph(5)

def main():
    print("Number of nodes")
    n = int(input())
    G.rand_vert_init(n)
    subtour = []
    for i in range(n):
        subtour.append(0)
        for j in range(i+1,n):
            G.edges.append([i, j])
    # for tau in range(G.tau):
    G.tour = []
    edges = list(G.edges)
    f = 0
    f_marginal = 0
    t = 0
    while edges!=[] and len(G.tour)<=n:
        max = 0
        p = -1
        fm = []
        for e in range(len(edges)):
            f_marginal, f = submodular_fun(G.tour, edges[e], f)
            fm.append(f_marginal)
        fm.sort()
        f_marginal = fm[0]
        if subtour[edges[p][0]]<2 and subtour[edges[p][1]]<2:
            G.tour.append(edges[p])
            subtour[edges[p][0]]+=1
            subtour[edges[p][1]]+=1
            ret = G.DFS()
            # print(ret)
            if ret == False and len(G.tour)<n:
                G.tour.pop(len(G.tour)-1)
                subtour[edges[p][0]]-=1
                subtour[edges[p][1]]-=1
        edges.pop(p)
    # print(G.tour)
    G.all_tour.append(G.tour)
    G.all_tour_costs.append(f)
    x = []
    y = []
    for e in G.tour:
        x = [G.points[e[0]][0], G.points[e[1]][0]]
        y = [G.points[e[0]][1], G.points[e[1]][1]]
        plt.plot(x, y)
    plt.show()
    G.tour = []

if __name__ == '__main__':
    main()
