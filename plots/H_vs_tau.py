import matplotlib
from mpl_toolkits import mplot3d
import csv
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import rc

H1 = []
H2 = []
H3 = []
H4 = []
H5 = []
H6 = []
H7 = []
H8 = []

with open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/H_vs_tau.csv', 'r') as file:
    reader = csv.reader(file)
    c = 0
    for row in reader:
        c+=1
        num=-1
        for r in row:
            temp = ''
            for t in r:
                if t!=';' and t!='\n' and t!='':
                    temp = temp + t
                else:
                    num+=1
                    temp = float(temp)
                    print(c, num)
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
                    temp = ''
x = []

matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.size'] = '13'
rc('text', usetex=True)
fig, ax = plt.subplots()
for i in range(0, 5000, 1):
    x.append(i)
x = np.array(x)
plt.ylim(-100, 200)
plt.xlim(0, 1000)

ysmoothed1 = gaussian_filter1d(H1, sigma=17)
ysmoothed2 = gaussian_filter1d(H2, sigma=17)
ysmoothed3 = gaussian_filter1d(H3, sigma=17)
ysmoothed4 = gaussian_filter1d(H4, sigma=17)
ysmoothed5 = gaussian_filter1d(H5, sigma=17)
ysmoothed6 = gaussian_filter1d(H6, sigma=17)
ysmoothed7 = gaussian_filter1d(H7, sigma=17)
# plt.plot(x, ysmoothed1, label=r"$\alpha = 0.1$")
plt.plot(x, ysmoothed2, label=r"\boldmath$\alpha = 0.1$")
plt.plot(x, ysmoothed3, label=r"\boldmath$\alpha = 0.3$")
plt.plot(x, ysmoothed4, label=r"\boldmath$\alpha = 0.5$")
plt.plot(x, ysmoothed5, label=r"\boldmath$\alpha = 0.7$")
plt.plot(x, ysmoothed6, label=r"\boldmath$\alpha = 0.9$")
plt.plot(x, ysmoothed7, label=r"\boldmath$\alpha = 1$")
ax.set_xlabel(r"$\tau$", fontsize=14)
ax.set_ylabel(r"$H(\mathcal{S}^{G}, \tau)$", fontsize=14)
ax.legend()
plt.show()
