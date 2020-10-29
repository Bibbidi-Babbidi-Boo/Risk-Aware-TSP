import matplotlib
from mpl_toolkits import mplot3d
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
from matplotlib import rc


H11 = []
H12 = []
H13 = []
H14 = []
H21 = []
H22 = []
H23 = []
H24 = []
H31 = []
H32 = []
H33 = []
H34 = []
H41 = []
H42 = []
H43 = []
H44 = []
H51 = []
H52 = []
H53 = []
H54 = []
H61 = []
H62 = []
H63 = []
H64 = []
H71 = []
H72 = []
H73 = []
H74 = []
temp = ''

with open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/runtime3.csv', 'r') as file:
    reader = csv.reader(file)
    c = 0
    for row in reader:
        c+=1
        num=-1
        for r in row:
            temp = ''
            temp = float(r)
            print(temp)
            # for t in r:
            #     if t!=';' and t!='\n' and t!='':
            #         temp = temp + t

                # else:
            num+=1
            temp = float(temp)
            # print(c, num, temp)
            if c==1 and num%4==0:
                H11.append(temp)
            if c==1 and num%4==1:
                H12.append(temp)
            if c==1 and num%4==2:
                H13.append(temp)
            if c==1 and num%4==3:
                H14.append(temp)
            if c==2 and num%4==0:
                H21.append(temp)
            if c==2 and num%4==1:
                H22.append(temp)
            if c==2 and num%4==2:
                H23.append(temp)
            if c==2 and num%4==3:
                H24.append(temp)
            if c==3 and num%4==0:
                H31.append(temp)
            if c==3 and num%4==1:
                H32.append(temp)
            if c==3 and num%4==2:
                H33.append(temp)
            if c==3 and num%4==3:
                H34.append(temp)
            if c==4 and num%4==0:
                H41.append(temp)
            if c==4 and num%4==1:
                H42.append(temp)
            if c==4 and num%4==2:
                H43.append(temp)
            if c==4 and num%4==3:
                H44.append(temp)
            if c==5 and num%4==0:
                H51.append(temp)
            if c==5 and num%4==1:
                H52.append(temp)
            if c==5 and num%4==2:
                H53.append(temp)
            if c==5 and num%4==3:
                H54.append(temp)
            if c==6 and num%4==0:
                H61.append(temp)
            if c==6 and num%4==1:
                H62.append(temp)
            if c==6 and num%4==2:
                H63.append(temp)
            if c==6 and num%4==3:
                H64.append(temp)
            if c==7 and num%4==0:
                H71.append(temp)
            if c==7 and num%4==1:
                H72.append(temp)
            if c==7 and num%4==2:
                H73.append(temp)
            if c==7 and num%4==3:
                H74.append(temp)
            temp = ''

t11 = 0
t12 = 0
t13 = 0
t14 = 0
t21 = 0
t22 = 0
t23 = 0
t24 = 0
t31 = 0
t32 = 0
t33 = 0
t34 = 0
t41 = 0
t42 = 0
t43 = 0
t44 = 0
t51 = 0
t52 = 0
t53 = 0
t54 = 0
t61 = 0
t62 = 0
t63 = 0
t64 = 0
t71 = 0
t72 = 0
t73 = 0
t74 = 0

for i in range(len(H11)):
    t11 += H11[i]/len(H11)
    t12 += H12[i]/len(H11)
    t13 += H13[i]/len(H11)
    t14 += H14[i]/len(H11)
    t21 += H21[i]/len(H11)
    t22 += H22[i]/len(H11)
    t23 += H23[i]/len(H11)
    t24 += H24[i]/len(H11)
    t31 += H31[i]/len(H11)
    t32 += H32[i]/len(H11)
    t33 += H33[i]/len(H11)
    t34 += H34[i]/len(H11)
    t41 += H41[i]/len(H11)
    t42 += H42[i]/len(H11)
    t43 += H43[i]/len(H11)
    t44 += H44[i]/len(H11)
    t51 += H51[i]/len(H11)
    t52 += H52[i]/len(H11)
    t53 += H53[i]/len(H11)
    t54 += H54[i]/len(H11)
    t61 += H61[i]/len(H11)
    t62 += H62[i]/len(H11)
    t63 += H63[i]/len(H11)
    t64 += H64[i]/len(H11)
print(t11)
# for i in range(len(H71)):
#     t71 += H71[i]/len(H71)
#     t72 += H72[i]/len(H71)
#     t73 += H73[i]/len(H71)
#     t74 += H74[i]/len(H71)

print(len(H11), len(H61))

x = np.array([5, 6, 7, 8, 9, 10])
y_0_1 = np.array([t11, t21, t31, t41, t51, t61])
y_0_1_err_min = np.array([(max(H11)+min(H11))/2, (max(H21)+min(H21))/2, (max(H31)+min(H31))/2, (max(H41)+min(H41))/2, (max(H51)+min(H51))/2, (max(H61)+min(H61))/2])
y_0_1_err_max = np.array([(max(H11)+min(H11))/2, (max(H21)+min(H21))/2, (max(H31)+min(H31))/2, (max(H41)+min(H41))/2, (max(H51)+min(H51))/2, (max(H61)+min(H61))/2])

y_0_5 = np.array([t12, t22, t32, t42, t52, t62])
y_0_5_err_min = np.array([(max(H12)+min(H12))/2, (max(H22)+min(H22))/2, (max(H32)+min(H32))/2, (max(H42)+min(H42))/2, (max(H52)+min(H52))/2, (max(H62)+min(H62))/2])
y_0_5_err_max = np.array([(max(H12)+min(H12))/2, (max(H22)+min(H22))/2, (max(H32)+min(H32))/2, (max(H42)+min(H42))/2, (max(H52)+min(H52))/2, (max(H62)+min(H62))/2])

y_0_9 = np.array([t13, t23, t33, t43, t53, t63])
y_0_9_err_min = np.array([(max(H13)+min(H13))/2, (max(H23)+min(H23))/2, (max(H33)+min(H33))/2, (max(H43)+min(H43))/2, (max(H53)+min(H53))/2, (max(H63)+min(H63))/2])
y_0_9_err_max = np.array([(max(H13)+min(H13))/2, (max(H23)+min(H23))/2, (max(H33)+min(H33))/2, (max(H43)+min(H43))/2, (max(H53)+min(H53))/2, (max(H63)+min(H63))/2])

# y_1 = np.array([t14, t24, t34, t44, t54, t64])
# y_1_err_min = np.array([min(H14), min(H24), min(H34), min(H44), min(H54), min(H64)])
# y_1_err_max = np.array([max(H14), max(H24), max(H34), max(H44), max(H54), max(H64)])


matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'
matplotlib.rcParams['font.size'] = '13'
rc('text', usetex=True)

fig, ax = plt.subplots()
# plt.ylim(0, 100)
plt.xlim(5, 10)
plt.plot(x, y_0_1, '-', color='tab:red')
plt.fill_between(x, y_0_1 - y_0_1_err_min, y_0_1 + y_0_1_err_max, color='tab:red', alpha=0.8, label=r"\boldmath$\alpha = 0.1$")
plt.plot(x, y_0_5, '-', color='tab:blue')
plt.fill_between(x, y_0_5 - y_0_5_err_min, y_0_5 + y_0_5_err_max, color='tab:blue', alpha=0.5, label=r"\boldmath$\alpha = 0.5$")
plt.plot(x, y_0_9, '-', color='tab:green')
plt.fill_between(x, y_0_9 - y_0_9_err_min, y_0_9 + y_0_9_err_max, color='tab:green', alpha=0.2, label=r"\boldmath$\alpha = 0.9$")
ax.set_xlabel(r"Number of vertices", fontsize=14)
ax.set_ylabel(r"Time (s)", fontsize=14)
ax.legend()
plt.show()
