import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import statistics

from information_map import Information_Map

file = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path/stochasatic_information_p(y)_vs_f2.csv', 'w')
file2 = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path/stochasatic_information_H_vs_alpha2.csv', 'w')

def best_edge_gain(e, f, Hf, reward, fail, tau, alpha, current_mean, current_var, edges, n, subtour, tour_temp):
    tour = list(tour_temp)
    tour.append(edges[e])
    mu = reward[e]
    sigma =  fail[e]
    sub = list(subtour)
    sub[edges[e][0]] += 1
    sub[edges[e][1]] += 1
    avg = 0
    var_avg = 0
    div = 0
    pos = []
    for j in range(len(tour)):
        p1 = tour[j][0]
        p2 = tour[j][1]
        for i in range(len(edges)):
            if sub[p1]<2:
                if (edges[i][0] == p1 or edges[i][1] == p1):
                    if edges[i][0] == p1:
                        if sub[edges[i][1]] < 2 and i not in pos:
                            pos.append(i)
                    else:
                        if sub[edges[i][0]] < 2 and i not in pos:
                            pos.append(i)
            if sub[p2]<2:
                if (edges[i][0] == p2 or edges[i][1] == p2):
                    if edges[i][0] == p2:
                        if sub[edges[i][1]] < 2 and i not in pos:
                            pos.append(i)
                    else:
                        if sub[edges[i][0]] < 2 and i not in pos:
                            pos.append(i)
    for i in pos:
        avg += reward[i]
        var_avg += fail[i]
    div = len(pos)
    if avg != 0:
        avg = avg/div
        var_avg = var_avg/div
    mu += current_mean
    sigma += current_var
    fUe = 0
    expectationfUe = 0
    ## Sample values of f(SUe)
    for i in range(300):
        sample = np.random.normal(mu, sqrt(sigma))
        t = sample + avg
        if tau-t>0:
            expectationfUe += tau-t
    expectationfUe /= 300
    HfUe = tau - expectationfUe/alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe

def main():
    ## For plotting table for different distributions
    for ta in range(1):
        ## Initialize information map
        M = Information_Map(0.0, 0.0)
        M.createInformation()
        M.points = []
        n = 8 ## Number of points
        M.rand_vert_init(n)
        M.plot()  ## Show map
        fig, ax = plt.subplots()
        max_reward = 0
        max_fail = 0
        ## Caluclate rewards and variances for every edge
        for i in range(n):
            for j in range(i+1,n):
                M.edges.append([i, j])
                po = M.drawLine(M.points[i], M.points[j])
                M.reward_calc(po)
        M.edge_failiure = []
        max_reward = statistics.median(M.edge_info_reward)
        for i in range(len(M.edge_info_reward)):
            M.edge_info_reward[i] = (M.edge_info_reward[i]*2/max_reward)**2
            M.edge_failiure.append(((M.edge_info_reward[i]+1)*3)**2)
        #     print(sqrt(M.edge_failiure[i]))
        print(M.edge_info_reward)
        ## For given alphas
        for alp in [0.01, 0.1, 0.3, 0.6, 0.9, 1]:
            M.alpha = alp
            H_max = -100000
            tau_max = 0
            tour_best = []
            ## For tau in range
            for tau in np.arange(0, 700, 1):
                ## Initialize Hf, f, H_marginal, subtour, reward, edges, fail(var)
                edges = list(M.edges)
                reward = list(M.edge_info_reward)
                fail = list(M.edge_failiure)
                subtour = [0]*n
                M.tau = float(tau)
                M.tour = []
                current_mean = 0
                current_var = 0
                print(M.tau, M.alpha, ta)
                Hf = M.tau-M.tau/M.alpha
                f = 0
                H_marginal = 0
                while edges!=[] and len(M.tour)<n:
                    Hm = []
                    for e in range(len(edges)):
                        H_marginal, HfUe = best_edge_gain(e, f, Hf, reward, fail, M.tau, M.alpha, current_mean, current_var, edges, n, subtour, M.tour)
                        Hm.append([H_marginal, HfUe, e])
                    Hm.sort(reverse=1)
                    new_edge = Hm[0][2]
                    ## Check degree constraint
                    if subtour[edges[new_edge][0]]<2 and subtour[edges[new_edge][1]]<2:
                        M.tour.append(edges[new_edge])
                        subtour[edges[new_edge][0]]+=1
                        subtour[edges[new_edge][1]]+=1
                        ret = M.DFS()
                        ## Check sub-loops
                        if ret == False and len(M.tour)<n:
                            M.tour.pop(len(M.tour)-1)
                            subtour[edges[new_edge][0]]-=1
                            subtour[edges[new_edge][1]]-=1
                        elif ret == True or len(M.tour) == n:
                            H_marginal = Hm[0][0]
                            Hf = Hm[0][1]
                            current_mean+=reward[new_edge]
                            current_var+=fail[new_edge]
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
            mu = 0
            sigma = 0
            for j in range(len(posn)):
                mu += M.edge_info_reward[posn[j]]
                sigma +=  M.edge_failiure[posn[j]]
                print(i, j, mu, sigma)
            for k in range(10000):
                f = np.random.normal(mu, sqrt(sigma))
                file.write('%f;' % float(f))
            file.write('\n')
        # M.plot()
        # ax.set_xlim(0, 99)
        # ax.set_ylim(0, 99)
        # for j in range(len(M.all_tour)):
        #     M.plot(M.all_tour[j])

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
