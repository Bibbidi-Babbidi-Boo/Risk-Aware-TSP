import matplotlib as plt
from mpl_toolkits import mplot3d
import csv
# import seaborn as sns
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

# tau_init = 20
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
temp = ''
alpha = 0.9

with open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path/stochasatic_information_H_vs_alpha.csv', 'r') as file:
    reader = csv.reader(file)
    c=0
    flag = 0
    for row in reader:
        c+=1
        num=-1
        for r in row:
            for t in r:
                if t!=';' and t!='\n' and t!='':
                    temp = temp + t
                else:
                    num+=1
                    temp = float(temp)
                    if c==1:
                        if flag==0:
                            H1.append(temp)
                        else:
                            H1[num] += temp
                    if c==2:
                        if flag==0:
                            H2.append(temp)
                        else:
                            H2[num] += temp
                    if c==3:
                        if flag==0:
                            H3.append(temp)
                        else:
                            H3[num] += temp
                    if c==4:
                        if flag==0:
                            H4.append(temp)
                        else:
                            H4[num] += temp
                    if c==5:
                        if flag==0:
                            H5.append(temp)
                        else:
                            H5[num] += temp
                    if c==6:
                        if flag==0:
                            H6.append(temp)
                        else:
                            H6[num] += temp
                    if c==7:
                        if flag==0:
                            H7.append(temp)
                        else:
                            H7[num] += temp
                    if c==8:
                        if flag==0:
                            H8.append(temp)
                        else:
                            H8[num] += temp
                    if c==9:
                        if flag==0:
                            H9.append(temp)
                        else:
                            H9[num] += temp
                    temp = ''
        if c== 9:
            c=0
            flag=1

print(H1)

tau_init = 0
tau = []
tau = [0.01, 0.1, 0.9, 0.99]
for i in range(len(H1)):
    # tau.append(tau_init)
    # tau_init+=100
    H1[i]/=1
    # H2[i]/=1
    # H3[i]/=1
    # H4[i]/=1
    # H5[i]/=1
    # H6[i]/=1
    # H7[i]/=1
    # H8[i]/=1
    # H9[i]/=1

print(len(H1), len(tau))

fig, ax = plt.subplots()

plt.scatter(tau, H1, marker='o')
# ax.plot(tau, H1, label = 'alpaha=0.1')
# ax.plot(tau, H2, label = 'alpaha=0.2')
# ax.plot(tau, H3, label = 'alpaha=0.3')
# ax.plot(tau, H4, label = 'alpaha=0.4')
# ax.plot(tau, H5, label = 'alpaha=0.5')
# ax.plot(tau, H6, label = 'alpaha=0.6')
# ax.plot(tau, H7, label = 'alpaha=0.7')
# ax.plot(tau, H8, label = 'alpaha=0.8')
# ax.plot(tau, H9, label = 'alpaha=0.9')

# clrs = sns.color_palette("husl", 9)
# for i in range(9):
#     means = []
#     if i==0:
#         avg = H1[0]
#         for j in range(len(H1)):
#             avg = (0.9*avg + 0.1*H1[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H1-means), dtype=np.float64)
#     if i==1:
#         avg = H2[0]
#         for j in range(len(H2)):
#             avg = (0.9*avg + 0.1*H2[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H2-means), dtype=np.float64)
#     if i==2:
#         avg = H3[0]
#         for j in range(len(H3)):
#             avg = (0.9*avg + 0.1*H3[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H3-means), dtype=np.float64)
#     if i==3:
#         avg = H4[0]
#         for j in range(len(H4)):
#             avg = (0.9*avg + 0.1*H4[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H4-means), dtype=np.float64)
#     if i==4:
#         avg = H5[0]
#         for j in range(len(H5)):
#             avg = (0.9*avg + 0.1*H5[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H5-means), dtype=np.float64)
#     if i==5:
#         avg = H6[0]
#         for j in range(len(H6)):
#             avg = (0.9*avg + 0.1*H6[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H6-means), dtype=np.float64)
#     if i==6:
#         avg = H7[0]
#         for j in range(len(H7)):
#             avg = (0.9*avg + 0.1*H7[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H7-means), dtype=np.float64)
#     if i==7:
#         avg = H8[0]
#         for j in range(len(H8)):
#             avg = (0.9*avg + 0.1*H8[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H8-means), dtype=np.float64)
#     if i==8:
#         avg = H9[0]
#         for j in range(len(H9)):
#             avg = (0.9*avg + 0.1*H9[j])
#             means.append(avg)
#         means = np.array(means)
#         sdt = np.array(abs(H9-means), dtype=np.float64)
#
#     with sns.axes_style("darkgrid"):
#         # ax.plot(tau, H1, label = 'alpaha=0.1')
#         ax.plot(tau, means, label = 'alpaha=0.'+str(i+1))
#         ax.fill_between(tau, means+sdt, means-sdt ,alpha=0.5)
#         ax.legend()

ax.legend()
# ax.set_xlim([10000,30000])
# ax.set_ylim([0,40])
plt.show()

#             for t in r:
#                 print(t)
#                 if t!=';' and t!='\n':
#                     temp = temp + t
#                     print("tem", temp)
#                 else:
#                     print("T", temp)
#                     H.append(float(temp))
#                     tau.append(tau_init)
#                     tau_init+=1
#                     temp = ''
#         ax.plot(tau, H, label = 'alpaha='+str(alpha))
#         H = []
#         tau = []
#         tau_init = 0
#         alpha+=0.1
#
# print(H)
# ax.legend()
# plt.show()
