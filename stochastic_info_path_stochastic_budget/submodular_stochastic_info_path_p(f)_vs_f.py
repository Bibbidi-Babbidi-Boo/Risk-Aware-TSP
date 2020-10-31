################################################################################

# This file contains the code used for comparing path selection performances of
# our algorithm for different values of alpha, beta, tau

################################################################################

import random
import numpy as np
from math import *
import matplotlib.pyplot as plt
import csv
import numpy as np
import statistics
from information_map import Information_Map
from scipy.stats import truncnorm
import os

def initialization(n):
    """
    Input: number of nodes n
    Output: The Map M, position of points on M, edges, reward and length of edges on the map
    """

    M = Information_Map(0.0, 0.0)
    M.createInformation()
    M.rand_vert_init(n)
    # M.plot()
    # exit()
    for i in range(n):
        print(i)
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
            po = M.drawLine(M.points[i], M.points[j])
            M.length_calc(po)
            pt = [(max(min(ceil(M.points[i][0]+2*cos(theta)), M.MAP_SIZE[0]-1), 0), max(min(ceil(M.points[i][1]+2*sin(theta)), M.MAP_SIZE[1]-1), 0)),
            (max(min(floor(M.points[i][0]-2*cos(theta)), M.MAP_SIZE[0]-1), 0), max(min(floor(M.points[i][1]-2*sin(theta)), M.MAP_SIZE[1]-1), 0)),
            (max(min(floor(M.points[j][0]-2*cos(theta)), M.MAP_SIZE[0]-1), 0), max(min(floor(M.points[j][1]-2*sin(theta)), M.MAP_SIZE[1]-1), 0)),
            (max(min(ceil(M.points[j][0]+2*cos(theta)), M.MAP_SIZE[0]-1), 0), max(min(ceil(M.points[j][1]+2*sin(theta)), M.MAP_SIZE[1]-1), 0))]
            po = M.rasterization(pt, i, j)
            M.reward_calc(po)
    ras_max = 0
    for i in range(len(M.edge_raster)):
        temp = 0
        for j in range(len(M.edge_raster[i])):
            temp+= M.map[M.edge_raster[i][j][0]][M.edge_raster[i][j][1]]
        if temp>ras_max:
            ras_max = temp
    max_reward = max(M.edge_reward)
    max_length = max(M.edge_length)
    for i in range(len(M.edge_reward)):
        M.edge_reward[i] = M.edge_reward[i]*10/ras_max
        M.edge_length[i] = M.edge_length[i]*10/max_length
    for i in range(M.MAP_SIZE[0]):
        for j in range(M.MAP_SIZE[1]):
            M.map[i][j] = M.map[i][j]*10/ras_max
    max_reward = max(M.edge_reward)
    max_length = max(M.edge_length)
    min_length = min(M.edge_length)

    return M

def record_results(beta_list, M, max_reward, min_reward, max_length, min_length, n):
    """
    Record the results to a csv file
    """

    file_name = '/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/20_points_3.csv'
    file = open(file_name, 'w')
    for i in range(len(M.all_tour)):
        print(i)
        M.tour = M.all_tour[i]
        temp = 0
        mu = M.gaussian_info[i][0]
        var = M.gaussian_info[i][1]
        mu_budget = M.gaussian_info[i][2]
        var_budget = M.gaussian_info[i][3]
        print(mu, var, mu_budget, var_budget)
        beta = beta_list[i%len(beta_list)]
        for i in range(10000):
            rew = np.random.normal(mu, sqrt(var))+100
            le = np.random.normal(mu_budget, sqrt(var_budget))+100
            while True:
                if le<0:
                    le = np.random.normal(mu_budget, sqrt(var_budget))
                else:
                    break
            while True:
                if rew<0:
                    rew = np.random.normal(mu, sqrt(var))
                else:
                    break
            t = (1-beta)*rew + beta*le
            file.write('%f;' % float(t))
        file.write('\n')
    file.close()
    fig, ax = plt.subplots()
    M.plot()
    ax.set_xlim(0, M.MAP_SIZE[0]-1)
    ax.set_ylim(0, M.MAP_SIZE[1]-1)
    for j in range(len(M.all_tour)):
        M.plot(M.best_points[j], M.all_tour[j])

def best_edge_gain(e, Hf, reward, edges, raster, length, tour_points, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, M, beta, max_reward, max_length, min_length):
    """
    Find the marginal gains of edge given

    Input: *edge e
    *current H
    *list of reward, points seen along each edge (raster), length of edges
    *already visited points tour_points
    *curren reward and cost mean and variance
    *map M
    *beta
    *max/min of reward, length

    Output: *H_marginal, H_new (marginal and new aux. function value)
    *mean/var of reward and length
    *points visited
    *util. function mean f
    """

    pt = list(tour_points)
    mu = current_mean_reward
    var =  current_var_reward
    mu_budget = current_mean_budget
    var_budget = current_var_budget
    temp = 0
    new_pt = raster[e]
    pos = []
    done_indices = []
    for i in range(len(new_pt)):
        if new_pt[i] not in pt:
            temp += M.map[new_pt[i][0]][new_pt[i][1]]
            pt.append(new_pt[i])
    temp2 = (max_length-length[e])
    mu += temp
    mu_budget += temp2
    var = (3*mu)**2
    var_budget = (3*mu_budget)**2
    expectationfUe = 0
    f = (1-beta)*mu + beta*mu_budget
    for i in range(100):
        rew = np.random.normal(mu, sqrt(var))+100
        le = np.random.normal(mu_budget, sqrt(var_budget))+100
        while True:
            if le<0:
                le = np.random.normal(mu_budget, sqrt(var_budget))
            else:
                break
        while True:
            if rew<0:
                rew = np.random.normal(mu, sqrt(var))
            else:
                break
        t = (1-beta)*rew + beta*le
        if M.tau-t>0:
            expectationfUe += M.tau-t
    expectationfUe /= 100
    HfUe = M.tau - expectationfUe/M.alpha
    H_marginal = HfUe - Hf
    return H_marginal, HfUe, mu, var, mu_budget, var_budget, pt, f

def get_best_greedy_path(M, beta, n, max_reward, min_reward, max_length, min_length):
    """
    Find tour with max. marginal gain

    Input: *map M
    *beta
    *no of vertices
    *max/min of reward and length

    Output: *H
    *mean/var of reward and length
    *mean of util function f
    *points visited
    """

    H_max = -100000
    mean_info = []
    Hf = M.tau-M.tau/M.alpha
    edges = list(M.edges)
    reward = list(M.edge_reward)
    raster = list(M.edge_raster)
    length = list(M.edge_length)
    position_holder_pop = 1000
    M.tour = []
    tour_points = []
    subtour = [0]*n
    ret = True
    current_mean_reward = 0
    current_var_reward = 0
    current_mean_budget = 0
    current_var_budget = 0
    print(M.tau, M.alpha, beta)
    while edges!=[] and len(M.tour)<n:
        if ret == True:
            Hm = []
            for e in range(len(edges)):
                H_marginal, HfUe, mu, var, mu_budget, var_budget, pt, f = best_edge_gain(e, Hf, reward, edges, raster, length, tour_points, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, M,  beta, max_reward, max_length, min_length)
                Hm.append([H_marginal, HfUe, e, mu, var, mu_budget, var_budget, pt, f])
            Hm.sort(reverse=1)
        else:
            for pos_iter in range(len(Hm)):
                if Hm[pos_iter][2] >= position_holder_pop:
                    Hm[pos_iter][2] -= 1
        new_edge = Hm[0][2]
        if subtour[edges[new_edge][0]]<2 and subtour[edges[new_edge][1]]<2:
            M.tour.append(edges[new_edge])
            subtour[edges[new_edge][0]]+=1
            subtour[edges[new_edge][1]]+=1
            ret = M.DFS()
            if ret == False and len(M.tour)<n:
                M.tour.pop(len(M.tour)-1)
                subtour[edges[new_edge][0]]-=1
                subtour[edges[new_edge][1]]-=1
                position_holder_pop = new_edge
            elif ret == True or len(M.tour) == n:
                H_marginal = Hm[0][0]
                Hf = Hm[0][1]
                current_mean_reward = Hm[0][3]
                current_var_reward = Hm[0][4]
                current_mean_budget = Hm[0][5]
                current_var_budget = Hm[0][6]
                tour_points = list(Hm[0][7])
                fUe = Hm[0][8]
                mean_info.append(fUe)
        else:
            ret = False
            position_holder_pop = new_edge
        edges.pop(new_edge)
        reward.pop(new_edge)
        raster.pop(new_edge)
        length.pop(new_edge)
        Hm.pop(0)
    return Hf, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, mean_info, tour_points

def main():
    """
    Main function running the algorithm for every value of alpha, beta, tau given, and finding the runtime
    """

    n = 20
    M = initialization(n)
    max_reward = max(M.edge_reward)
    min_reward = min(M.edge_reward)
    max_length = max(M.edge_length)
    min_length = min(M.edge_length)
    for alp in [0.1]:
        M.alpha = alp
        beta_list = [0, 1]
        tour_best = []
        all_points = []
        best_info = []
        for beta in beta_list:
            H_max = -100000
            for tau in np.arange(0, 10000, 1):
                M.tau = float(tau)
                Hf, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, mean_info, tour_points = get_best_greedy_path(M, beta, n, max_reward, min_reward, max_length, min_length)
                print(Hf, current_mean_reward, current_mean_budget)
                if Hf>H_max:
                    best_info = list(mean_info)
                    all_points = list(tour_points)
                    H_max = Hf
                    tour_best = list(M.tour)
                    info = [current_mean_reward+100, current_var_reward, current_mean_budget+100, current_var_budget]
                if Hf < -100:
                    break
            print("BEST", best_info)
            M.all_tour.append(tour_best)
            M.best_points.append(all_points)
            M.gaussian_info.append(info)
            print("GASS", M.gaussian_info)
    record_results(beta_list, M, max_reward, min_reward, max_length, min_length, n)

if __name__ == '__main__':
    main()
