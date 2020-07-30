import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import *
import random

from information_map import Information_Map

file = open('risk_distance_H_v_tau.csv', 'w')

def best_edge_gain(e, f, Hf, reward, fail, tau, alpha):
    mu = reward[e]
    sigma =  fail[e]
    # print(reward)
    # print(fail)
    # exit()
    fUe = 0
    expectationfUe = 0
    sample = -10
    for i in range(100):
        sample = -100
        while True:
            sample = np.random.normal(mu, sigma)
            if mu/10<sample<10*mu:
                break
        t = f + sample
        if tau-t>0:
            expectationfUe += tau-t
        else:
            pass
        fUe += t
    expectationfUe /= 100
    fUe /= 100
    HfUe = tau - expectationfUe/alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe, fUe

def main():
    M = Information_Map(150, 0.5)
    M.createInformation()
    n = 5
    M.rand_vert_init(n)
    for i in range(n):
        for j in range(i+1,n):
            M.edges.append([i, j])
            M.drawLine(M.points[i], M.points[j])
    subtour = [0]*n
    edges = list(M.edges)
    reward = list(M.edge_info_reward)
    fail = list(M.edge_failiure)
    Hf = M.tau*(1-1/M.alpha)
    f = 0
    H_marginal = 0
    H_max = -10
    while edges!=[] and len(M.tour)<=n:
        Hm = []
        for e in range(len(edges)):
            H_marginal, HfUe, fUe = best_edge_gain(e, f, Hf, reward, fail, M.tau, M.alpha)
            Hm.append([H_marginal, HfUe, fUe, e])
        Hm.sort(reverse=1)
        new_edge = Hm[0][3]
        if subtour[edges[new_edge][0]]<2 and subtour[edges[new_edge][1]]<2:
            M.tour.append(edges[new_edge])
            subtour[edges[new_edge][0]]+=1
            subtour[edges[new_edge][1]]+=1
            ret = M.DFS()
            if ret == False and len(M.tour)<n:
                M.tour.pop(len(M.tour)-1)
                subtour[edges[new_edge][0]]-=1
                subtour[edges[new_edge][1]]-=1
            else:
                H_marginal = Hm[0][0]
                Hf = Hm[0][1]
                f = Hm[0][2]
        edges.pop(new_edge)
        fail.pop(new_edge)
        reward.pop(new_edge)
    expect = M.f_calc()
    M.alpha = 0.0
    for o in range(9):
        M.alpha += 0.1
        for tau1 in range(0, int(max(expect)), 1):
            tau = float(tau1)
            H = 0
            for i in range(len(expect)):
                if expect[i]<tau:
                    H += tau-expect[i]
                else:
                    H+=0
            H /= len(expect)
            H = tau - H/(M.alpha)
            print("H", H, tau)
            file.write('%f;' % float(H))
        file.write('\n')
    # M.plot()

    # G.alpha = 0.0
    # G.points = []
    # G.rand_vert_init(n)
    # subtour = [0]*n
    # edges = list(G.edges)
    # f = 0
    # f_marginal = 0
    # while edges!=[] and len(G.tour)<=n:
    #     p = -1
    #     fm = []
    #     for e in range(len(edges)):
    #         f_marginal, fUe = find_good_path(edges[e], f)
    #         fm.append([f_marginal, fUe, e])


if __name__ == '__main__':
    main()
