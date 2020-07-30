import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np

from information_map import Information_Map

file = open('risk_distance_p(y)_vs_f.csv', 'w')
file2 = open('risk_distance_H_vs_alpha.csv', 'w')

def best_edge_gain(e, f, Hf, reward, fail, tau, alpha):
    mu = reward[e]
    sigma =  fail[e]
    # print(reward)
    # print(fail)
    # exit()
    fUe = 0
    expectationfUe = 0
    for i in range(100):
        sample = -100
        sample = np.random.normal(mu, sigma)
        # if sample<0:
        #     sample = 0
        # while True:
        sample = np.random.normal(mu, sigma)
            # if 0<sample<2*mu:
                # break
        t = f + sample
        # print(sample, mu)
        if tau-t>0:
            expectationfUe += tau-t
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
    n = 10
    M.rand_vert_init(n)
    M.plot()
    fig, ax = plt.subplots()
    for i in range(len(M.points)):
        plt.scatter(M.points[i][0], M.points[i][1], marker='o', color='b')
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 99)
    plt.show()
    for i in range(n):
        for j in range(i+1,n):
            M.edges.append([i, j])
            po = M.drawLine(M.points[i], M.points[j])
            M.reward_calc(po)
    # m = max(M.edge_info_reward)
    # for i in range(len(M.edge_info_reward)):
    #     M.edge_info_reward[i] = M.edge_info_reward[i]/m
    #     M.edge_failiure[i] = M.edge_failiure[i]/(m)
    subtour = [0]*n
    for i in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        M.alpha = i
        H_max = -1000
        tau_max = 0
        tour_best = []
        for tau in np.arange(0, 300, 1):
            edges = list(M.edges)
            reward = list(M.edge_info_reward)
            fail = list(M.edge_failiure)
            subtour = [0]*n
            M.tau = float(tau)
            M.tour = []
            print(M.tau, M.alpha)
            Hf = M.tau-M.tau/M.alpha
            f = 0
            H_marginal = 0
            while edges!=[] and len(M.tour)<=n:
                Hm = []
                for e in range(len(edges)):
                    # print("EE", e)
                    H_marginal, HfUe, fUe = best_edge_gain(e, f, Hf, reward, fail, M.tau, M.alpha)
                    Hm.append([H_marginal, HfUe, fUe, e])
                Hm.sort(reverse=1)
                # exit()
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
            print(Hf)
            if Hf>H_max:
                tau_max = M.tau
                H_max = Hf
                tour_best = list(M.tour)
        M.all_tour.append(tour_best)
        file2.write('%f;' % float(H_max))


    print(M.all_tour)
    for i in range(len(M.all_tour)):
        print(i)
        M.tour = M.all_tour[i]
        temp = 0
        posn = []
        for l in range(len(M.tour)):
            for j in range(len(M.edges)):
                if M.edges[j][0] == M.tour[l][0] and M.edges[j][1] == M.tour[l][1]:
                    posn.append(j)
                    break
        for k in range(1000):
            f = 0
            for j in range(len(posn)):
                mu = M.edge_info_reward[posn[j]]
                sigma =  M.edge_failiure[posn[j]]
                sample = -100
                # sample = np.random.normal(mu, sigma)
                # if sample<0:
                #     sample = 0
                # while True:
                sample = np.random.normal(mu, sigma)
                    # if 0<sample<2*mu:
                        # break
                f += sample
            file.write('%f;' % float(f))
        file.write('\n')
    M.plot()
    ax.set_xlim(0, 99)
    ax.set_ylim(0, 99)
    for j in range(len(M.all_tour)):
        M.plot(M.all_tour[j])

    # for j in range(len(M.all_tour)):
    #     for i in range(len(M.points)):
    #         plt.scatter(M.points[i][0], M.points[i][1], marker='o', color='b')
    #     M.tour = M.all_tour[j]
    #     for i in range(len(M.tour)):
    #         x1 = [M.points[M.tour[i][0]][0], M.points[M.tour[i][1]][0]]
    #         y1 = [M.points[M.tour[i][0]][1], M.points[M.tour[i][1]][1]]
    #         plt.plot(x1, y1, color='r')
    #     plt.show()

if __name__ == '__main__':
    main()
