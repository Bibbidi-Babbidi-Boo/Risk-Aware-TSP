import matplotlib
from mpl_toolkits import mplot3d
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from matplotlib import rc

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
H12 = []
H13 = []
H14 = []
H15 = []
H16 = []
H17 = []
H18 = []
H19 = []
H20 = []
H21 = []
H22 = []
H23 = []
H24 = []
H25 = []
H26 = []
H27 = []
H28 = []
H29 = []
H30 = []
H31 = []
H32 = []
H33 = []
H34 = []
H35 = []
H36 = []
H37 = []
H38 = []
H39 = []
H40 = []
H41 = []
H42 = []
H43 = []
H44 = []
H45 = []
H46 = []
H47 = []
H48 = []
H49 = []
H50 = []
H51 = []
H52 = []
H53 = []
H54 = []
H55 = []
H56 = []
H57 = []
H58 = []
H59 = []
H60 = []
H61 = []
H62 = []
H63 = []
H64 = []
H65 = []
H66 = []
H67 = []
H68 = []
H69 = []
H70 = []
H71 = []
H72 = []
H73 = []
H74 = []
H75 = []
H76 = []
H77 = []
H78 = []
H79 = []
H80 = []
H81 = []
H82 = []
H83 = []
H84 = []
H85 = []
H86 = []
H87 = []
H88 = []
H89 = []
H90 = []
H91 = []
H92 = []
H93 = []
H94 = []
H95 = []
H96 = []
H97 = []
H98 = []
H99 = []
H100 = []
H101 = []
H102 = []
H103 = []
H104 = []
H105 = []
H106 = []
H107 = []
H108 = []
H109 = []
H110 = []
H111 = []
H112 = []
H113 = []
H114 = []
H115 = []
H116 = []
H117 = []
H118 = []
H119 = []
H120 = []
H121 = []
H122 = []
temp = ''

with open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/submodular_vs_baseline_final_v2_1.csv', 'r') as file:
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
                    if c==12:
                        H12.append(temp)
                    if c==13:
                        H13.append(temp)
                    if c==14:
                        H14.append(temp)
                    if c==15:
                        H15.append(temp)
                    if c==16:
                        H16.append(temp)
                    if c==17:
                        H17.append(temp)
                    if c==18:
                        H18.append(temp)
                    if c==19:
                        H19.append(temp)
                    if c==20:
                        H20.append(temp)
                    if c==21:
                        H21.append(temp)
                    if c==22:
                        H22.append(temp)
                    if c==23:
                        H23.append(temp)
                    if c==24:
                        H24.append(temp)
                    if c==25:
                        H25.append(temp)
                    if c==26:
                        H26.append(temp)
                    if c==27:
                        H27.append(temp)
                    if c==28:
                        H28.append(temp)
                    if c==29:
                        H29.append(temp)
                    if c==30:
                        H30.append(temp)
                    if c==31:
                        H31.append(temp)
                    if c==32:
                        H32.append(temp)
                    if c==33:
                        H33.append(temp)
                    if c==34:
                        H34.append(temp)
                    if c==35:
                        H35.append(temp)
                    if c==36:
                        H36.append(temp)
                    if c==37:
                        H37.append(temp)
                    if c==38:
                        H38.append(temp)
                    if c==39:
                        H39.append(temp)
                    if c==40:
                        H40.append(temp)
                    if c==41:
                        H41.append(temp)
                    if c==42:
                        H42.append(temp)
                    if c==43:
                        H43.append(temp)
                    if c==44:
                        H44.append(temp)
                    if c==45:
                        H45.append(temp)
                    if c==46:
                        H46.append(temp)
                    if c==47:
                        H47.append(temp)
                    if c==48:
                        H48.append(temp)
                    if c==49:
                        H49.append(temp)
                    if c==50:
                        H50.append(temp)
                    if c==51:
                        H51.append(temp)
                    if c==52:
                        H52.append(temp)
                    if c==53:
                        H53.append(temp)
                    if c==54:
                        H54.append(temp)
                    if c==55:
                        H55.append(temp)
                    if c==56:
                        H56.append(temp)
                    if c==57:
                        H57.append(temp)
                    if c==58:
                        H58.append(temp)
                    if c==59:
                        H59.append(temp)
                    if c==60:
                        H60.append(temp)
                    if c==61:
                        H61.append(temp)
                    if c==62:
                        H62.append(temp)
                    if c==63:
                        H63.append(temp)
                    if c==64:
                        H64.append(temp)
                    if c==65:
                        H65.append(temp)
                    if c==66:
                        H66.append(temp)
                    if c==67:
                        H67.append(temp)
                    if c==68:
                        H68.append(temp)
                    if c==69:
                        H69.append(temp)
                    if c==70:
                        H70.append(temp)
                    if c==71:
                        H71.append(temp)
                    if c==72:
                        H72.append(temp)
                    if c==73:
                        H73.append(temp)
                    if c==74:
                        H74.append(temp)
                    if c==75:
                        H75.append(temp)
                    if c==76:
                        H76.append(temp)
                    if c==77:
                        H77.append(temp)
                    if c==78:
                        H78.append(temp)
                    if c==79:
                        H79.append(temp)
                    if c==80:
                        H80.append(temp)
                    if c==81:
                        H81.append(temp)
                    if c==82:
                        H82.append(temp)
                    if c==83:
                        H83.append(temp)
                    if c==84:
                        H84.append(temp)
                    if c==85:
                        H85.append(temp)
                    if c==86:
                        H86.append(temp)
                    if c==87:
                        H87.append(temp)
                    if c==88:
                        H88.append(temp)
                    if c==89:
                        H89.append(temp)
                    if c==90:
                        H90.append(temp)
                    if c==91:
                        H91.append(temp)
                    if c==92:
                        H92.append(temp)
                    if c==93:
                        H93.append(temp)
                    if c==94:
                        H94.append(temp)
                    if c==95:
                        H95.append(temp)
                    if c==96:
                        H96.append(temp)
                    if c==97:
                        H97.append(temp)
                    if c==98:
                        H98.append(temp)
                    if c==99:
                        H99.append(temp)
                    if c==100:
                        H100.append(temp)
                    if c==101:
                        H101.append(temp)
                    if c==102:
                        H102.append(temp)
                    if c==103:
                        H103.append(temp)
                    if c==104:
                        H104.append(temp)
                    if c==105:
                        H105.append(temp)
                    if c==106:
                        H106.append(temp)
                    if c==107:
                        H107.append(temp)
                    if c==108:
                        H108.append(temp)
                    if c==109:
                        H109.append(temp)
                    if c==110:
                        H110.append(temp)
                    if c==111:
                        H111.append(temp)
                    if c==112:
                        H112.append(temp)
                    if c==113:
                        H113.append(temp)
                    if c==114:
                        H114.append(temp)
                    if c==115:
                        H115.append(temp)
                    if c==116:
                        H116.append(temp)
                    if c==117:
                        H117.append(temp)
                    if c==118:
                        H118.append(temp)
                    if c==119:
                        H119.append(temp)
                    if c==120:
                        H120.append(temp)
                    if c==121:
                        H121.append(temp)
                    temp = ''

matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.size'] = '13'
rc('text', usetex=True)

fig, ax = plt.subplots()
xs = np.linspace(150, 350)

density = gaussian_kde(H1)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.01$", color='tab:blue')
m = max(density(xs))

density = gaussian_kde(H12)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.1$", color='tab:orange')
m1 = max(density(xs))
if m1>m:
    m = m1

# density = gaussian_kde(H113)
# density._compute_covariance()
# ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.2$", color='tab:orange')
# m1 = max(density(xs))
# if m1>m:
#     m = m1


density = gaussian_kde(H34)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.3$", color='tab:green')
m1 = max(density(xs))
if m1>m:
    m = m1

# density = gaussian_kde(H115)
# density._compute_covariance()
# ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.4$", color='tab:green')
# m1 = max(density(xs))
# if m1>m:
#     m = m1


density = gaussian_kde(H56)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.5$", color='tab:red')
m1 = max(density(xs))
if m1>m:
    m = m1

# density = gaussian_kde(H117)
# density._compute_covariance()
# ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.6$", color='tab:red')
# m1 = max(density(xs))
# if m1>m:
#     m = m1

density = gaussian_kde(H78)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.7$", color='tab:purple')
m1 = max(density(xs))
if m1>m:
    m = m1

# density = gaussian_kde(H119)
# density._compute_covariance()
# ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.8$", color='tab:purple')
# m1 = max(density(xs))
# if m1>m:
#     m = m1

density = gaussian_kde(H100)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=0.9$", color='tab:brown')
m1 = max(density(xs))
if m1>m:
    m = m1

density = gaussian_kde(H111)
density._compute_covariance()
ax.plot(xs,density(xs),label=r"\boldmath$\alpha=1$", color='tab:pink')
m1 = max(density(xs))
if m1>m:
    m = m1


ax.set_xlabel(r"Value of the utility function $f(\mathcal{S}^{G}, y)$", fontsize=15)
ax.set_ylabel(r"Probability distribution of $f(\mathcal{S}^{G}, y)$", fontsize=15)#, fontsize=14

x_mean = np.array([221.73705764979982, 221.65506884210123, 238.8638478636004, 241.61082341610057, 243.3101345237997, 245.24792546629925, 243.61691167750018])
x_std = np.array([25.255, 25, 45.985, 47.813, 49.919, 53.386, 51.7017])
y_mean = np.array([m+0.007, m+0.006, m+0.005, m+0.004, m+0.003, m+0.002, m+0.001])
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
for i in range(len(colors)):
    ax.errorbar(x_mean[i], y_mean[i], xerr=x_std[i], fmt='o', capsize=4, markersize=4, color = colors[i])
fig.subplots_adjust(left=0.15, bottom=0.14, top=0.986, right=0.986, wspace=None, hspace=None)

plt.legend()
plt.show()
