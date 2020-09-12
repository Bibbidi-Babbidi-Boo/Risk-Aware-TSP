import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import statistics

from information_map import Information_Map

def best_edge_gain(e, Hf, reward, fail, length, tau, alpha, current_mean, current_var, current_mean_budget, beta, n, subtour, tour_temp, max_edge_reward, edges, sum_of_largest_edges):
    tour = list(tour_temp)
    tour.append(edges[e])
    mu = reward[e]
    var =  fail[e]
    sub = list(subtour)
    sub[edges[e][0]] += 1
    sub[edges[e][1]] += 1
    sum_reward = 0
    avg_len = 0
    var_reward = 0
    div = 0
    pos = []
    done_indices = []
    ## Find which edges end in open vertices (degree <2)
    for j in range(len(tour)):
        p1 = tour[j][0]
        p2 = tour[j][1]
        for i in range(len(edges)):
            if p1 not in done_indices and sub[p1]<2:
                done_indices.append(p1)
                if (edges[i][0] == p1 or edges[i][1] == p1):
                    if edges[i][0] == p1 and edges[i][1] != p2:
                        if sub[edges[i][1]] < 2 and i not in pos:
                            pos.append(i)
                    elif edges[i][1] == p1 and edges[i][0] != p2:
                        if sub[edges[i][0]] < 2 and i not in pos:
                            pos.append(i)
            if p2 not in done_indices and sub[p2]<2:
                done_indices.append(p2)
                if (edges[i][0] == p2 or edges[i][1] == p2):
                    if edges[i][0] == p2 and edges[i][1] != p1:
                        if sub[edges[i][1]] < 2 and i not in pos:
                            pos.append(i)
                    elif edges[i][1] == p2 and edges[i][0] != p1:
                        if sub[edges[i][0]] < 2 and i not in pos:
                            pos.append(i)
    for i in pos:
        sum_reward += reward[i]
        avg_len += length[i]
        var_reward += fail[i]
    # if avg != 0:
    #     avg = avg/div
    #     avg_len = avg_len/div
    #     var_avg = var_avg/div
    mu += current_mean
    var += current_var
    fUe = 0
    expectationfUe = 0
    ## Sample values of f(SUe)
    for i in range(500):
        t = np.random.normal(mu+sum_reward, sqrt(var+var_reward)) +  max_edge_reward*len(tour)
        t = (1-beta)*t + beta*(sum_of_largest_edges*len(tour) - (current_mean_budget + length[e] + avg_len))
        if tau-t>0:
            expectationfUe += tau-t
        fUe += t
    expectationfUe /= (500)
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
        n = 6 ## Number of points
        M.rand_vert_init(n)
        M.plot()  ## Show map
        fig, ax = plt.subplots()
        max_fail = 0
        ## Caluclate rewards and variances for every edge
        for i in range(n):
            for j in range(i+1,n):
                M.edges.append([i, j])
                po = M.drawLine(M.points[i], M.points[j])
                M.reward_calc(po)
        M.edge_failiure = []
        ## Normalise the reward and variances to a similar scale
        median_edge_length = statistics.median(M.edge_length)
        median_reward = statistics.median(M.edge_info_reward)
        for i in range(len(M.edge_info_reward)):
            M.edge_info_reward[i] = (M.edge_info_reward[i]*5/median_reward)**2
            M.edge_failiure.append((M.edge_info_reward[i]*3)**2)
            M.edge_length[i] = (M.edge_length[i]*5/median_edge_length)**2
        max_edge_length = max(M.edge_length)
        max_edge_reward = max(M.edge_info_reward)
        for i in range(len(M.edge_length)):
            M.edge_length[i] = M.edge_length[i]*max_edge_reward/max_edge_length
        max_edge_length = max(M.edge_length)
        max_edge_reward = max(M.edge_info_reward)
        ## Find the sum of n largest edges
        new = list(M.edge_info_reward)
        new.sort(reverse=True)
        largest_n_edge = new[0:n-1]
        sum_of_largest_edges = sum(largest_n_edge)
        ## For given alphas
        file_name = '/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_deterministic_budget/stochasatic_information_p(y)_vs_f_expectation_alpha_vs_beta=0.5.csv'
        file = open(file_name, 'w')
        for alp in [0.01, 0.1, 0.3, 0.6, 0.9, 1]:
            ## Save file
            # file_name = '/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_deterministic_budget/stochasatic_information_p(y)_vs_f_expectation'+str(alp)+'.csv'
            # file = open(file_name, 'w')
            M.alpha = alp
            H_max = -100000
            tour_best = []
            M.tour = []
            for beta in [0]:
                ## For every tau in range
                for tau in np.arange(0, 6000, 5):
                    ## Initialize Hf, H_marginal, subtour(to check subtours after adding an edge), reward(for edges), edges, fail(var), ret
                    M.tau = float(tau)
                    Hf = M.tau-M.tau/M.alpha
                    H_marginal = 0
                    edges = list(M.edges)
                    reward = list(M.edge_info_reward)
                    fail = list(M.edge_failiure)
                    length = list(M.edge_length)
                    subtour = [0]*n
                    position_holder_pop = 1000
                    M.tour = []
                    ret = True
                    ## Reset the values of the current mean and variance of the path selected so far
                    current_mean = 0
                    current_var = 0
                    current_mean_budget = 0
                    print(M.tau, M.alpha, beta)
                    ## While a tour is not formed
                    while edges!=[] and len(M.tour)<n:
                        ## If the tour so far is valid take new samples to find the next best edge
                        if ret == True:
                            Hm = []
                            for e in range(len(edges)):
                                H_marginal, HfUe = best_edge_gain(e, Hf, reward, fail, length, M.tau, M.alpha, current_mean, current_var, current_mean_budget, beta, n, subtour, M.tour, max_edge_reward, edges, sum_of_largest_edges)
                                Hm.append([H_marginal, HfUe, e])
                            Hm.sort(reverse=1)
                        ## If the tour is invalid, remove use the previous sampled information
                        else:
                            for pos_iter in range(len(Hm)):
                                if Hm[pos_iter][2] >= position_holder_pop:
                                    Hm[pos_iter][2] -= 1
                        new_edge = Hm[0][2]
                        ## Check degree constraint of tour formed
                        if subtour[edges[new_edge][0]]<2 and subtour[edges[new_edge][1]]<2:
                            M.tour.append(edges[new_edge])
                            subtour[edges[new_edge][0]]+=1
                            subtour[edges[new_edge][1]]+=1
                            ret = M.DFS()
                            ## Check sub-loops. If the loop is bad (ret=False), remove the new added edge, and change the edge numbers in Hm, so it can be used again
                            if ret == False and len(M.tour)<n:
                                M.tour.pop(len(M.tour)-1)
                                subtour[edges[new_edge][0]]-=1
                                subtour[edges[new_edge][1]]-=1
                                position_holder_pop = new_edge
                            ## If the tour is good, retain the edge
                            elif ret == True or len(M.tour) == n:
                                H_marginal = Hm[0][0]
                                Hf = Hm[0][1]
                                current_mean += reward[new_edge]
                                current_var += fail[new_edge]
                                current_mean_budget += length[new_edge]
                        else:
                            ret = False
                            position_holder_pop = new_edge
                        ## Update the list of edges, rewards and fail
                        edges.pop(new_edge)
                        fail.pop(new_edge)
                        reward.pop(new_edge)
                        length.pop(new_edge)
                        Hm.pop(0)
                    print(Hf)
                    ## If the new tour formed is better than previous tours, retain it
                    if Hf>H_max:
                        H_max = Hf
                        tour_best = list(M.tour)
                M.all_tour.append(tour_best)

            ##For each tour
            for i in range(len(M.all_tour)):
                print(i)
                M.tour = M.all_tour[i]
                temp = 0
                posn = []
                ## Find positions of the edges used
                for l in range(len(M.tour)):
                    for j in range(len(M.edges)):
                        if M.edges[j][0] == M.tour[l][0] and M.edges[j][1] == M.tour[l][1]:
                            posn.append(j)
                            break
                mu = 0
                var = 0
                ## For each edge used
                for j in range(len(posn)):
                    mu += M.edge_info_reward[posn[j]]
                    var +=  M.edge_failiure[posn[j]]
                    print(i, j, mu, var)
                mu += max_edge_reward*n
                for k in range(10000):
                    f = np.random.normal(mu, sqrt(var))
                    file.write('%f;' % float(f))
                file.write('\n')
        file.close()
        M.plot()
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 99)
        for j in range(len(M.all_tour)):
            M.plot(M.all_tour[j])

if __name__ == '__main__':
    main()
