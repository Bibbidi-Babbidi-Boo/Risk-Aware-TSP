import matplotlib as plt
from mpl_toolkits import mplot3d
import csv
# import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

tau = []
H1 = []
H2 = []
H3 = []
H4 = []
H5 = []
H6 = []
H7 = []
H8 = []
H9 = []
H10 = []
H11 = []
temp = ''

with open('risk_distance_p(y)_vs_f.csv', 'r') as file:
    reader = csv.reader(file)
    c = 0
    for row in reader:
        c+=1
        num=0
        for r in row:
            temp = ''
            num+=1
            for t in r:
                if t!=';' and t!='\n' and t!='':
                    temp = temp + t
                else:
                    # print("C", c)
                    temp = float(temp)
                    # temp = int(temp)
                    # print("Temp", temp)
                    if c==1:
                        H1.append(temp)
                    if c==2:
                        H2.append(temp)
                    if c==3:
                        H3.append(temp)
                    if c==4:
                        H4.append(temp)
                    if c==5:
                        H5.append(temp)
                    if c==6:
                        H6.append(temp)
                    if c==7:
                        H7.append(temp)
                    if c==8:
                        H8.append(temp)
                    if c==9:
                        H9.append(temp)
                    if c==10:
                        H10.append(temp)
                    if c==11:
                        H11.append(temp)
                        # tau.append(num)
                    temp = ''

# print(H3)
# print(max(H1), max(H2), max(H3), max(H4), max(H5), max(H6), max(H7), max(H8), max(H9))
fig, ax = plt.subplots()
xs = np.linspace(100,1500)
#
# density = gaussian_kde(H1)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.01')

density = gaussian_kde(H2)
density._compute_covariance()
ax.plot(xs,density(xs),label='alpha=0.1')
# # #
# density = gaussian_kde(H3)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.2')
# # # # #
density = gaussian_kde(H4)
density._compute_covariance()
ax.plot(xs,density(xs),label='alpha=0.3')
# # # # #
# density = gaussian_kde(H5)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.4')
# # # #
# density = gaussian_kde(H6)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.5')
# # # #
# density = gaussian_kde(H7)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.6')
# # # # #
density = gaussian_kde(H8)
density._compute_covariance()
ax.plot(xs,density(xs),label='alpha=0.7')
# # #
# density = gaussian_kde(H9)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.8')
#
density = gaussian_kde(H10)
density._compute_covariance()
ax.plot(xs,density(xs),label='alpha=0.9')

# density = gaussian_kde(H11)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='alpha=0.99')
# print(density(xs))

# plt.xlim([200, 400])
# plt.ylim([0, 0.04])
# ax.hist(H1, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.01')
# ax.hist(H2, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.1')
# ax.hist(H3, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.2')
# ax.hist(H4, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.3')
# ax.hist(H5, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.4')
# ax.hist(H6, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.5')
# ax.hist(H7, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.6')
# ax.hist(H8, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.7')
# ax.hist(H9, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.8')
# ax.hist(H10, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.9')
# ax.hist(H11, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.99')

plt.legend()
plt.show()

# data = pd.read_csv('risk_distance_H_v_tau.csv', sep=';',header=None, index_col =0)
# data = pd.DataFrame(data=data)
# print(data)
# # data=data.astype(float).transpose
# # # print(data)
# #
# data.plot(kind='bar')
# # plt.ylabel('Frequency')
# # plt.xlabel('Words')
# # plt.title('Title')

plt.show()
