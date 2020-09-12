import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import statistics

from information_map import Information_Map
## Saving files for plots
file = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path/stochasatic_information_p(y)_vs_f.csv', 'w')
file2 = open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path/stochasatic_information_H_vs_alpha.csv', 'w')

def best_edge_gain(e, Hf, reward, edges, raster, tour_points, current_mean, current_var, M, subtour, max_reward):
    pt = list(tour_points)
    mu = current_mean
    var = current_var
    temp = 0
    new_pt = raster[e]
    pos = []
    done_indices = []
    for i in range(len(new_pt)):
        if new_pt[i] not in pt:
            temp += M.map[new_pt[i][0]][new_pt[i][1]]
            pt.append(new_pt[i])
    mu += temp
    var += (3*temp)**2
    fUe = 0
    expectationfUe = 0
    ## Sample values of f(SUe)
    for i in range(1000):
        t = np.random.normal(mu, sqrt(var))
        if M.tau-t>0:
            expectationfUe += M.tau-t
        fUe += t
    expectationfUe /= (1000)
    HfUe = M.tau - expectationfUe/M.alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe, mu, var, pt

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
                M.edges.append([i, j])
                if (M.points[j][0]-M.points[i][0]) == 0:
                    theta = 0
                else:
                    slope = (M.points[j][1]-M.points[i][1])/(M.points[j][0]-M.points[i][0])
                    if slope == 0:
                        theta = np.pi/2
                    else:
                        theta = atan(-1/slope)
                # pt = [(10, 10), (10, 80), (30, 80), (30, 10)]
                pt = [(max(min(ceil(M.points[i][0]+2*cos(theta)), 99), 0), max(min(ceil(M.points[i][1]+2*sin(theta)), 99), 0)),
                (max(min(floor(M.points[i][0]-2*cos(theta)), 99), 0), max(min(floor(M.points[i][1]-2*sin(theta)), 99), 0)),
                (max(min(floor(M.points[j][0]-2*cos(theta)), 99), 0), max(min(floor(M.points[j][1]-2*sin(theta)), 99), 0)),
                (max(min(ceil(M.points[j][0]+2*cos(theta)), 99), 0), max(min(ceil(M.points[j][1]+2*sin(theta)), 99), 0))]
                po = M.raster(pt, i, j)
                M.reward_calc(po)

        ## Normalise the reward and variances to a similar scale
        max_reward = max(M.edge_info_reward)
        for i in range(M.MAP_SIZE[0]):
            for j in range(M.MAP_SIZE[1]):
                M.map[i][j] /= max_reward
        for i in range(len(M.edge_info_reward)):
            M.edge_info_reward[i] /= max_reward
        max_reward = max(M.edge_info_reward)
        gaussian_info = []
        all_mean_info = []
        ## For given alphas
        for alp in [0.01, 0.1, 0.3, 0.6, 0.9, 1]:
            M.alpha = alp
            H_max = -100000
            tour_best = []
            all_points = []
            all_means = []
            ## For every tau in range
            for tau in np.arange(0, 1000, 1):
                means = []
                ## Initialize Hf, H_marginal, subtour(to check subtours after adding an edge), reward(for edges), edges, fail(var), ret
                M.tau = float(tau)
                Hf = M.tau-M.tau/M.alpha
                H_marginal = 0
                edges = list(M.edges)
                reward = list(M.edge_info_reward)
                raster = list(M.edge_raster)
                tour_points = []
                subtour = [0]*n
                position_holder_pop = 1000
                M.tour = []
                ret = True
                ## Reset the values of the current mean and variance of the path selected so far
                current_mean = 0
                current_var = 0
                print(M.tau, M.alpha, ta)
                ## While a tour is not formed
                while edges!=[] and len(M.tour)<n:
                    ## If the tour so far is valid take new samples to find the next best edge
                    if ret == True:
                        Hm = []
                        for e in range(len(edges)):
                            H_marginal, HfUe, mu, var, pt = best_edge_gain(e, Hf, reward, edges, raster, tour_points, current_mean, current_var, M, subtour, max_reward)
                            Hm.append([H_marginal, HfUe, e, mu, var, pt])
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
                            current_mean = Hm[0][3]
                            current_var = Hm[0][4]
                            tour_points = list(Hm[0][5])
                            means.append(current_mean)
                    else:
                        ret = False
                        position_holder_pop = new_edge
                    ## Update the list of edges, rewards and fail
                    edges.pop(new_edge)
                    reward.pop(new_edge)
                    raster.pop(new_edge)
                    Hm.pop(0)
                print(Hf, current_mean)
                if Hf < -8/(5*M.alpha):
                    break
                ## If the new tour formed is better than previous tours, retain it
                if Hf>H_max:
                    all_means = list(means)
                    all_points = list(tour_points)
                    H_max = Hf
                    tour_best = list(M.tour)
                    info = [current_mean, current_var]
            ## Save tours for all alphas
            all_mean_info.append(all_means)
            M.all_tour.append(tour_best)
            M.best_points.append(all_points)
            gaussian_info.append(info)
            # file2.write('%f;' % float(H_max))

        ##For each tour
        for i in range(len(all_mean_info)):
            for j in range(len(all_mean_info[i])):
                print(j, all_mean_info[i][j])
            print("------------")
        for i in range(len(M.all_tour)):
            print(i)
            M.tour = M.all_tour[i]
            temp = 0
            posn = []
            ## Find positions of the edges used
            mu = gaussian_info[i][0]
            var = gaussian_info[i][1]
            print(mu, var)
            ## For each edge used
            for k in range(10000):
                f = np.random.normal(mu, sqrt(var))
                file.write('%f;' % float(f))
            file.write('\n')
        M.plot()
        ax.set_xlim(0, 99)
        ax.set_ylim(0, 99)
        for j in range(len(M.all_tour)):
            M.plot(M.best_points[j], M.all_tour[j])
if __name__ == '__main__':
    main()
