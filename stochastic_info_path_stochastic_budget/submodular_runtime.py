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
import time

def initialization(n):
    M = Information_Map(0.0, 0.0)
    M.createInformation()
    M.rand_vert_init(n)
    # M.plot()
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
            po = M.drawLine(M.points[i], M.points[j])
            M.edge_length.append(exp(0.02*len(po)))
            # M.length_calc(po)
            pt = [(max(min(ceil(M.points[i][0]+0*cos(theta)), 99), 0), max(min(ceil(M.points[i][1]+0*sin(theta)), 99), 0)),
            (max(min(floor(M.points[i][0]-0*cos(theta)), 99), 0), max(min(floor(M.points[i][1]-0*sin(theta)), 99), 0)),
            (max(min(floor(M.points[j][0]-0*cos(theta)), 99), 0), max(min(floor(M.points[j][1]-0*sin(theta)), 99), 0)),
            (max(min(ceil(M.points[j][0]+0*cos(theta)), 99), 0), max(min(ceil(M.points[j][1]+0*sin(theta)), 99), 0))]
            po = M.rasterization(pt, i, j)
            # reward = 0
            # for p in po:
                # reward += 1/(1+exp(-7.5*M.map[p[0]][p[1]]/1+3))
            # M.edge_reward.append(reward)
            M.reward_calc(po)
    ras_max = 0
    for i in range(len(M.edge_raster)):
        temp = 0
        for j in range(len(M.edge_raster[i])):
            # temp += 1/(1+exp(-7.5*M.map[M.edge_raster[i][j][0]][M.edge_raster[i][j][1]]/1+3))
            # temp+= exp(1*M.map[M.edge_raster[i][j][0]][M.edge_raster[i][j][1]])
            temp+= M.map[M.edge_raster[i][j][0]][M.edge_raster[i][j][1]]
        if temp>ras_max:
            ras_max = temp
    max_reward = max(M.edge_reward)
    max_length = max(M.edge_length)
    for i in range(len(M.edge_reward)):
        M.edge_reward[i] = M.edge_reward[i]*10/ras_max
        # M.edge_reward[i] = 1000/(1+exp(-7.5*M.edge_reward[i]/1000+3))
        M.edge_length[i] = M.edge_length[i]*10/max_length
    for i in range(M.MAP_SIZE[0]):
        for j in range(M.MAP_SIZE[1]):
            # M.map[i][j] = M.map[i][j]*1000/(ras_max*max(M.edge_reward))
            # M.map[i][j] = 1000/(ras_max*(1+exp(-7.5*M.map[i][j]/1+3)))
            M.map[i][j] = M.map[i][j]*10/ras_max
            # M.map[i][j] = exp(1*M.map[i][j])*1000/ras_max
    max_length = max(M.edge_length)
    min_length = min(M.edge_length)
    l_temp = 0
    r_temp = 0
    for i in range(len(M.edge_raster)):
        temp = 0
        l_temp += max_length-M.edge_length[i]
        r_temp += M.edge_reward[i]
        for j in range(len(M.edge_raster[i])):
            temp+= M.map[M.edge_raster[i][j][0]][M.edge_raster[i][j][1]]
        # print(temp, M.edge_reward[i], max_length-M.edge_length[i])
    # print("F", r_temp, l_temp)
    # exit()
    return M

def record_results(start, end, file):
    val = end-start
    print(val)
    file.write('%f;' % float(val))

def best_edge_gain(e, Hf, reward, edges, raster, length, tour_points, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, M, beta, max_reward, max_length, min_length):
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
    var = (1.5*mu)**2
    # print(var)
    var_budget += (1.5*mu_budget)**2
    expectationfUe = 0
    f = (1-beta)*mu + beta*mu_budget
    # print("AA", ((max(2*mu-(len(M.tour)+1)*max_reward, 0)), (min((len(M.tour)+1)*max_reward, 2*mu))))
    # if max(2*mu-(len(M.tour)+1)*max_reward, 0)-mu >= min((len(M.tour)+1)*max_reward, 2*mu)-mu:
    #     rew = [mu]*100
    # rew = truncnorm.rvs((0-mu)/sqrt(var), ((len(M.tour)+1)*max_reward-mu)/sqrt(var), size=100)*sqrt(var)+mu
    # le = truncnorm.rvs((0-mu_budget)/sqrt(var_budget+0.0001), ((len(M.tour)+1)*max_length-mu_budget)/sqrt(var_budget+0.0001), size=100)*sqrt(var_budget)+mu_budget
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
    # print(M.tau, M.alpha, beta)
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
    file_name = '/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/runtime.csv'
    file = open(file_name, 'w')
    alpha_val = [0.1, 0.5, 0.9]
    for n in np.arange(4,10,1):
        for i in range(100):
            print("start", n)
            M = initialization(n)
            max_reward = max(M.edge_reward)
            min_reward = min(M.edge_reward)
            max_length = max(M.edge_length)
            min_length = min(M.edge_length)
            start = time.time()
            tau_prev = 0
            for alp in alpha_val:
                M.alpha = alp
                tour_best = []
                all_points = []
                best_info = []
                beta = int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1)
                H_max = -100000
                for tau in np.arange(50, 200, 1):
                    M.tau = float(tau)
                    Hf, current_mean_reward, current_var_reward, current_mean_budget, current_var_budget, mean_info, tour_points = get_best_greedy_path(M, beta, n, max_reward, min_reward, max_length, min_length)
                    print(Hf, M.tau)
                    if Hf>H_max:
                        tau_prev = M.tau
                        best_info = list(mean_info)
                        all_points = list(tour_points)
                        H_max = Hf
                        tour_best = list(M.tour)
                    if H_max-Hf>H_max/2:
                        break
                end = time.time()
                print("end")
                record_results(start, end, file)
        file.write('\n')

if __name__ == '__main__':
    main()
