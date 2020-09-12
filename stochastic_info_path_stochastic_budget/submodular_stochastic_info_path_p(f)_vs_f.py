import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import statistics

from information_map import Information_Map

def best_edge_gain(e, Hf, reward, edges, raster, length, tour_points, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, M, beta, max_reward, max_edge_length, current_edges, current_lengths):
    ## Current points in the tour, their mean and variance
    pt = list(tour_points)
    mu = current_mean_reward
    var =  current_var_reward
    temp = 0
    ## Rasterization of new edge
    new_pt = raster[e]
    pos = []
    done_indices = []
    ## Add points only from new edge if theyre not covered previously
    for i in range(len(new_pt)):
        if new_pt[i] not in pt:
            temp += M.map[new_pt[i][0]][new_pt[i][1]]
            pt.append(new_pt[i])
    ## Increase the mean and variance
    mu += temp
    var += (3*temp)**2
    ## Increase the new budget
    mu_budget = current_mean_budget+length[e]
    var_budget = current_var_budget+(0.3*(max_edge_length-length[e]+0.001))**2
    expectationfUe = 0
    ## Define the submodular function f
    f = (1-beta)*mu + beta*((len(M.tour)+1)*max_edge_length - mu_budget)
    ## Sample values of f
    for i in range(1000):
        rew = np.random.normal(mu, sqrt(var))
        le = np.random.normal(mu_budget, sqrt(var_budget))
        while True:
            if le<0:
                le = np.random.normal(mu_budget, sqrt(var_budget))
            else:
                break
        t = (1-beta)*rew + beta*((len(M.tour)+1)*max_edge_length - le)
        if M.tau-t>0:
            expectationfUe += M.tau-t
    ## Find the expectation, H and return values
    expectationfUe /= (1000)
    HfUe = M.tau - expectationfUe/M.alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe, mu, var, mu_budget, var_budget, pt, f

def main():
    ## For plotting table for different distributions
    for ta in range(1):
        ## Initialize information map, with n vertices
        M = Information_Map(0.0, 0.0)
        M.createInformation()
        M.points = []
        n = 8 ## Number of points
        M.rand_vert_init(n)
        M.plot()  ## Show map
        fig, ax = plt.subplots()
        ## Caluclate rewards and variances for every edge
        for i in range(n):
            for j in range(i+1,n):
                ## Calculate the prpendicular to each line to give the line thickness
                M.edges.append([i, j])
                if (M.points[j][0]-M.points[i][0]) == 0:
                    theta = 0
                else:
                    slope = (M.points[j][1]-M.points[i][1])/(M.points[j][0]-M.points[i][0])
                    if slope == 0:
                        theta = np.pi/2
                    else:
                        theta = atan(-1/slope)
                ## Draw the line and find all points connecting the two vertices
                po = M.drawLine(M.points[i], M.points[j])
                M.length_calc(po)
                ## Give the end points of the line with thickness, which forms a rectangle (change the x in x*cos/sin)
                pt = [(max(min(ceil(M.points[i][0]+0*cos(theta)), 99), 0), max(min(ceil(M.points[i][1]+0*sin(theta)), 99), 0)),
                (max(min(floor(M.points[i][0]-0*cos(theta)), 99), 0), max(min(floor(M.points[i][1]-0*sin(theta)), 99), 0)),
                (max(min(floor(M.points[j][0]-0*cos(theta)), 99), 0), max(min(floor(M.points[j][1]-0*sin(theta)), 99), 0)),
                (max(min(ceil(M.points[j][0]+0*cos(theta)), 99), 0), max(min(ceil(M.points[j][1]+0*sin(theta)), 99), 0))]
                ## Calculate the rasterization of these points
                po = M.rasterization(pt, i, j)
                ## Calculate the new rewards given the thick lines
                M.reward_calc(po)
        ## Normalise the reward and variances to 1
        max_reward = max(M.edge_info_reward)
        max_length = max(M.edge_length)
        for i in range(M.MAP_SIZE[0]):
            for j in range(M.MAP_SIZE[1]):
                M.map[i][j] /= max_reward
        for i in range(len(M.edge_info_reward)):
            M.edge_info_reward[i] /= max_reward
            M.edge_length[i] /= max_length
        max_reward = max(M.edge_info_reward)
        max_length = max(M.edge_length)
        ## Save file name
        file_name = '/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/stochasatic_information_p(y)_vs_f_alpha_vs_beta=0.5_function.csv'
        file = open(file_name, 'w')
        ## For all alphas
        for alp in [0.01, 0.1, 0.3, 0.6, 0.9, 1]:
            ## Initialize alpha, H_max, tour_best(which holds the edges of the best tour), all_points(which holds all the points of the best tour), all_means_fUe(which holds the fUe for the best tour of any alpha)
            M.alpha = alp
            H_max = -100000
            tour_best = []
            all_points = []
            all_means_fUe = []
            ## For the specified beta values
            for beta in [0.5]:
                # For every tau in range
                for tau in np.arange(0, 200, 0.1):
                    ## Initialize Hf, H_marginal, subtour(to check subtours after adding an edge), reward(for edges), edges, fail(var), ret, mean_fUe(holds the values of fUe for each tour), current_edges(holds the edges added so far), current_lengths(length of edges so far)
                    mean_fUe = []
                    current_edges = []
                    current_lengths = []
                    M.tau = float(tau)
                    Hf = M.tau-M.tau/M.alpha
                    H_marginal = 0
                    edges = list(M.edges)
                    reward = list(M.edge_info_reward)
                    raster = list(M.edge_raster)
                    length = list(M.edge_length)
                    tour_points = []
                    subtour = [0]*n
                    position_holder_pop = 1000
                    M.tour = []
                    ret = True
                    ## Reset the values of the current mean and variance of the path selected so far
                    current_mean_reward = 0
                    current_var_reward = 0
                    current_mean_budget = 0
                    current_var_budget = 0
                    fUe = 0
                    print(M.tau, M.alpha, beta)
                    ## While a tour is not formed
                    while edges!=[] and len(M.tour)<n:
                        ## If the tour so far is valid take new samples to find the next best edge
                        if ret == True:
                            Hm = []
                            for e in range(len(edges)):
                                H_marginal, HfUe, mu, var, mu_budget, var_budget, pt, f = best_edge_gain(e, Hf, reward, edges, raster, length, tour_points, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, M,  beta, max_reward, max_length, current_edges, current_lengths)
                                Hm.append([H_marginal, HfUe, e, mu, var, mu_budget, var_budget, pt, f])
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
                                current_mean_reward = Hm[0][3]
                                current_var_reward = Hm[0][4]
                                current_mean_budget = Hm[0][5]
                                current_var_budget = Hm[0][6]
                                tour_points = list(Hm[0][7])
                                fUe = Hm[0][8]
                                mean_fUe.append(fUe)
                        else:
                            ret = False
                            position_holder_pop = new_edge
                        ## Update the list of edges, rewards and fail
                        current_edges.append(reward[new_edge])
                        current_lengths.append(length[new_edge])
                        edges.pop(new_edge)
                        reward.pop(new_edge)
                        raster.pop(new_edge)
                        length.pop(new_edge)
                        Hm.pop(0)
                    print(Hf, current_mean_reward, current_mean_budget)
                    ## If too low reward: break
                    if Hf < -8/(5*M.alpha):
                        break
                    ## If the new tour formed is better than previous tours, retain it
                    if Hf>H_max:
                        all_means_fUe = list(mean_fUe)
                        all_points = list(tour_points)
                        H_max = Hf
                        tour_best = list(M.tour)
                        info = [current_mean_reward, current_var_reward, current_mean_budget, current_var_budget]
                ## Save tours for all alphas
                M.all_fUe.append(all_means_fUe)
                M.all_tour.append(tour_best)
                M.best_points.append(all_points)
                M.gaussian_info.append(info)
                print("GASS", M.gaussian_info)

        ##For each tour
        for i in range(len(M.all_fUe)):
            for j in range(len(M.all_fUe[i])):
                print(j, M.all_fUe[i][j])
            print("------------")
        # for i in range(len(M.gaussian_info)):
        #     print(M.gaussian_info[i][0])
        # print(M.all_tour)
        for i in range(len(M.all_tour)):
            print(i)
            M.tour = M.all_tour[i]
            temp = 0
            ## Find positions of the edges used
            mu = M.gaussian_info[i][0]
            var = M.gaussian_info[i][1]
            mu_budget = M.gaussian_info[i][2]
            var_budget = M.gaussian_info[i][3]
            print(mu, var, mu_budget, var_budget)
            beta = 0.5
            ## For each edge used
            for k in range(10000):
                rew = np.random.normal(mu, sqrt(var))
                le = np.random.normal(mu_budget, sqrt(var_budget))
                while True:
                    if le<0:
                        le = np.random.normal(mu_budget, sqrt(var_budget))
                    else:
                        break
                t = (1-beta)*rew + beta*(len(M.tour)*max_length - le)
                file.write('%f;' % float(t))
            file.write('\n')
        file.close()
        M.plot()
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 99)
        for j in range(len(M.all_tour)):
            M.plot(M.best_points[j], M.all_tour[j])

if __name__ == '__main__':
    main()
