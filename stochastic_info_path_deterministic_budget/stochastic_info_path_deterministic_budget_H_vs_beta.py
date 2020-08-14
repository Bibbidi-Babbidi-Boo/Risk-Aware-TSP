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

def best_edge_gain(e, f, Hf, reward, fail, length, tau, alpha, current_mean, current_var, current_length, beta):
    global cur_max, max_length
    mu = reward[e]
    sigma =  fail[e]
    l = length[e]
    count = 1
    if length[e]>max_length:
        max_length = length[e]
    for i in range(len(current_mean)):
        count += 1
        mu += current_mean[i]
        sigma += current_var[i]
        l += length[i]
    fUe = 0
    expectationfUe = 0
    ## Sample values of f(SUe)
    for i in range(100):
        sample = np.random.normal(mu, sigma)
        # if sample>cur_max:
        #     cur_max = sample
        cur_max = mu
        t = sample
        if tau - ((1-beta)*t/cur_max + beta*(count*max_length - l)/max_length)>0:
            expectationfUe += tau - ((1-beta)*t/cur_max + beta*(count*max_length - l)/max_length)
        else:
            expectationfUe += 0
        fUe += t
    expectationfUe /= 100
    fUe /= 100
    HfUe = tau - expectationfUe/alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe, fUe

def main():
    M = Information_Map(0.0, 0.0)
    M.createInformation()
    M.points = []
    n = 10 ## Number of points
    M.rand_vert_init(n)
    M.plot()  ## Show map
    fig, ax = plt.subplots()
    for i in range(n):
        for j in range(i+1,n):
            M.edges.append([i, j])
            po = M.drawLine(M.points[i], M.points[j])
            M.edge_length.append(len(po))
            M.reward_calc(po)
    print(M.edge_info_reward)
    subtour = [0]*n
    edges = list(M.edges)
    reward = list(M.edge_info_reward)
    fail = list(M.edge_failiure)
    length = list(M.edge_length)
    subtour = [0]*n
    M.tau = float(tau)
    M.tour = []
    current_mean = []
    current_var = []
    current_length = []
    print(M.tau, M.alpha, beta)
    Hf = M.tau-M.tau/M.alpha
    f = 0
    H_marginal = 0
    while edges!=[] and len(M.tour)<n:
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
    expect = []
    posn = []
    mu = 0
    sigma = 0
    for i in range(len(M.tour)):
        for j in range(len(M.edges)):
            if M.edges[j][0] == M.tour[i][0] and M.edges[j][1] == M.tour[i][1]:
                posn.append(j)
                break
    for j in range(len(posn)):
        mu += M.edge_info_reward[posn[j]]
        sigma +=  M.edge_failiure[posn[j]]
    for k in range(10000):
        sample = np.random.normal(mu, sigma)
        expect.append(f)

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

if __name__ == '__main__':
    main()
