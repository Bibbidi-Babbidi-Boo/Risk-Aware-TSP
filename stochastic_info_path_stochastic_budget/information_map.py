################################################################################

# This script contains code to generate the map M for the simulations

################################################################################

#!/usr/bin/env python
import time
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import *
import random
import numpy
from PIL import Image, ImageDraw
from matplotlib.path import Path

class Information_Map:
    """
    Class to generate the map
    """

    def __init__(self, tau, alpha):
        """
        Initialization of map size, num of distributions, and all other values as 0/[]
        """

        self.MAP_SIZE = (2000, 2000)
        self.map = np.zeros((self.MAP_SIZE[0], self.MAP_SIZE[1]))
        self.variance_scale = [self.MAP_SIZE[0], self.MAP_SIZE[1]]
        self.NUM_DISTRIBUTIONS = 20
        self.MAX_VAL = random.sample(range(800, 1000), self.NUM_DISTRIBUTIONS)
        self.points = []
        self.edges = []
        self.edge_reward = []
        self.edge_raster = []
        self.edge_length = []
        self.tau = tau
        self.alpha = alpha
        self.tour = []
        self.all_tour = []
        self.best_points = []
        self.all_fUe = []
        self.gaussian_info = []

    def rasterization(self, poly, n1, n2):
        """
        To rasterize line between two points into discrete set of points
        Refer Bresenham's algorithm

        Input: Two end points n1, n2

        Output: Set of points forming the approximate line between the two end
        points
        """

        img = Image.new('1', (self.MAP_SIZE[1], self.MAP_SIZE[0]), 0)
        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
        mask = numpy.array(img)
        mask = np.transpose(mask)
        temp = []
        for i in range(self.MAP_SIZE[0]):
            for j in range(self.MAP_SIZE[1]):
                if mask[i][j] == 1:
                    temp.append([i, j])
        po = self.drawLine(self.points[n1], self.points[n2])
        for i in range(len(po)):
            if po[i] not in temp:
                temp.append(po[i])
        self.edge_raster.append(temp)
        return temp

    def rand_vert_init(self, n):
        """
        Randomly initialize n vertices on the map
        """

        while len(self.points) < n:
            x = random.randint(0,self.MAP_SIZE[0]-1)
            y = random.randint(0,self.MAP_SIZE[1]-1)
            p = [x,y]
            count = 0
            if p not in self.points:
                for point in self.points:
                    if sqrt((point[0]-p[0])**2 + (point[1]-p[1])**2)>8+3*len(self.points):
                        count +=1
            if count == len(self.points):
                self.points.append(p)
            else:
                pass

    def bivariateGaussianMatrix(self, pos):
        """
        Initialize the gaussian distributions on the map to show distribution of
        reward
        """

        gaussian_mean = [min(self.MAP_SIZE[0]-5, max(5, self.MAP_SIZE[0]*np.random.rand())), min(self.MAP_SIZE[1]-5, max(5, self.MAP_SIZE[1]*np.random.rand()))]
        gaussian_var = np.zeros(2)
        gaussian_var[0] = self.variance_scale[0]*random.sample(range(int(20*self.MAP_SIZE[0]/100), int(50*self.MAP_SIZE[0]/100)), 1)[0]/60
        gaussian_var[1] = self.variance_scale[1]*random.sample(range(int(20*self.MAP_SIZE[1]/100), int(50*self.MAP_SIZE[1]/100)), 1)[0]/60
        SigmaX = np.sqrt(gaussian_var[0])
        SigmaY = np.sqrt(gaussian_var[1])
        for i in range(self.MAP_SIZE[0]):
            for j in range(self.MAP_SIZE[1]):
                self.map[i][j] += self.MAX_VAL[pos]*(1/(2*np.pi*SigmaX*SigmaY))*exp(-((i-gaussian_mean[0])**2)/(2*SigmaX**2) -((j-gaussian_mean[1])**2)/(2*SigmaY**2))

    def createInformation(self):
        """
        Sum the bivariate gaussians to form map
        """
        for i in range(self.NUM_DISTRIBUTIONS):
            self.bivariateGaussianMatrix(i)
            print(i)
        max_value = max(list(map(max, self.map)))

    def plot(self, edge_points = [], array = []):
        """
        Plot map an the tours formed
        """

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.MAP_SIZE[0]-1)
        ax.set_ylim(0, self.MAP_SIZE[1]-1)
        for p in edge_points:
            plt.scatter(p[0], p[1], s=3, color='k')
        for i in range(len(self.points)):
            plt.scatter(self.points[i][0],self.points[i][1], s=15, color='w')
        if array != []:
            for i in range(len(array)):
                po = self.drawLine(self.points[array[i][0]], self.points[array[i][1]])
                for p in po:
                    plt.scatter(p[0], p[1], s=3, color='r')
        plt.imshow(self.map)
        plt.colorbar()
        plt.show()

    def drawLine(self, start, end):
        '''
        Implements Bresenham's line algorithm
        From http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
        '''
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1
        dy = y2 - y1
        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx
        if swapped:
            points.reverse()
        return points

    def reward_calc(self, points):
        """
        Calculating reward for given edge
        """

        reward = 0
        for p in points:
            reward += self.map[p[0]][p[1]]
        self.edge_reward.append(reward)

    def length_calc(self, points):
        """
        Calculating length of given edge
        """

        self.edge_length.append(len(points))

    def DFS(self):
        """
        Depth first search to check for subtours and degree invalidificaiton
        """
        visited = [False]*len(self.points)
        path = list(self.tour)
        edge = [-100, -100]
        start = 0
        ret = True
        while ret == True and path != []:
            i = path[0]
            edge = i
            start = i[0]
            vertex1 = i[0]
            vertex2 = i[1]
            ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
        return ret

    def recursive_next(self, edge, path, visited, start, vertex1, vertex2):
        """
        To recursively run the search algorithm for subtour finding
        """
        visited[vertex1] = True
        flag = 0
        path.remove(edge)
        for j in range(len(path)):
            if path[j][0] == vertex2 or path[j][1] == vertex2:
                flag+=1
                if path[j][0] == vertex2:
                    vertex1 = path[j][0]
                    vertex2 = path[j][1]
                else:
                    vertex1 = path[j][1]
                    vertex2 = path[j][0]
                edge = path[j]
                break
        if vertex2 == start:
            return False
        if flag == 0:
            return True
        ret = self.recursive_next(edge, path, visited, start, vertex1, vertex2)
        return ret
