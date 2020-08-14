import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np

from information_map import Information_Map

global cur_max, max_length
cur_max = 0
max_length = 0

# file = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_deterministic_budget/stochasatic_information_deterministic_budget_p(y)_vs_f.csv', 'w')
file2 = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_deterministic_budget/stochasatic_information_deterministic_budget_H(beta)_vs_beta.csv', 'w')

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
        # print("T", t)
        # print("INFO", ((1-beta)*t/cur_max + beta*(count*cur_max - length[e])/max_length))
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
    ## Initialize information map
    M = Information_Map(0.0, 0.0)
    M.createInformation()
    M.points = []
    n = 5 ## Number of points
    M.rand_vert_init(n)
    M.plot()  ## Show map
    fig, ax = plt.subplots()
    ## Caluclate rewards and variances for every edge
    for i in range(n):
        for j in range(i+1,n):
            M.edges.append([i, j])
            po = M.drawLine(M.points[i], M.points[j])
            M.edge_length.append(len(po))
            M.reward_calc(po)
    print(M.edge_info_reward)
    subtour = [0]*n
    ## For given alphas
    for i in [0.01, 0.99]:
        for temp_beta in range(0, 10):
            beta = temp_beta/10
            M.alpha = i
            H_max = -100000
            tau_max = 0
            tour_best = []
            ## For tau in range
            for tau in np.arange(0, 200, 1):
                ## Initialize Hf, f, H_marginal, subtour, reward, edges, fail(var)
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
                        H_marginal, HfUe, fUe = best_edge_gain(e, f, Hf, reward, fail, length, M.tau, M.alpha, current_mean, current_var, current_length, beta)
                        Hm.append([H_marginal, HfUe, fUe, e])
                    Hm.sort(reverse=1)
                    new_edge = Hm[0][3]
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
                            current_mean.append(reward[new_edge])
                            current_var.append(fail[new_edge])
                            current_length.append(length[e])
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
        file2.write('\n')









    # print(M.all_tour)
    # for i in range(len(M.all_tour)):
    #     print(i)
    #     M.tour = M.all_tour[i]
    #     temp = 0
    #     posn = []
    #     for l in range(len(M.tour)):
    #         for j in range(len(M.edges)):
    #             if M.edges[j][0] == M.tour[l][0] and M.edges[j][1] == M.tour[l][1]:
    #                 posn.append(j)
    #                 break
    #     mu = 0
    #     sigma = 0
    #     for j in range(len(posn)):
    #         mu += M.edge_info_reward[posn[j]]
    #         sigma +=  M.edge_failiure[posn[j]]
    #         for k in range(10000):
    #             while True:
    #                 f = np.random.normal(mu, sigma)
    #                 if 0<f<2*mu:
    #                     break
    #             # sample = np.random.normal(mu, sigma)
    #             # if sample<0:
    #             #     sample = 0
    #             # while True:
    #             # f = np.random.normal(mu, sigma)
    #                 # if 0<sample<2*mu:
    #                     # break
    #             file.write('%f;' % float(f))
    #         file.write('\n')
    # M.plot()
    # ax.set_xlim(0, 99)
    # ax.set_ylim(0, 99)
    # for j in range(len(M.all_tour)):
    #     M.plot(M.all_tour[j])
    #
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
