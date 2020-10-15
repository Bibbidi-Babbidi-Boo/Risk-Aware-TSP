import matplotlib
from mpl_toolkits import mplot3d
import csv
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

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

with open('/home/rishab/Risk-Aware-TSP/plots/stochastic_info_path_stochastic_budget/submodular_vs_baseline4.csv', 'r') as file:
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

for c in np.arange(1,125):
    val = 0
    for j in range(len(H1)):
        if c==1:
            val += H1[j]
        if c==2:
            val += H2[j]
        if c==3:
            val += H3[j]
        if c==4:
            val += H4[j]
        if c==5:
            val += H5[j]
        if c==6:
            val += H6[j]
        if c==7:
            val += H7[j]
        if c==8:
            val += H8[j]
        if c==9:
            val += H9[j]
        if c==10:
            val += H10[j]
        if c==11:
            val += H11[j]
        if c==12:
            val += H12[j]
        if c==13:
            val += H13[j]
        if c==14:
            val += H14[j]
        if c==15:
            val += H15[j]
        if c==16:
            val += H16[j]
        if c==17:
            val += H17[j]
        if c==18:
            val += H18[j]
        if c==19:
            val += H19[j]
        if c==20:
            val += H20[j]
        if c==21:
            val += H21[j]
        if c==22:
            val += H22[j]
        if c==23:
            val += H23[j]
        if c==24:
            val += H24[j]
        if c==25:
            val += H25[j]
        if c==26:
            val += H26[j]
        if c==27:
            val += H27[j]
        if c==28:
            val += H28[j]
        if c==29:
            val += H29[j]
        if c==30:
            val += H30[j]
        if c==31:
            val += H31[j]
        if c==32:
            val += H32[j]
        if c==33:
            val += H33[j]
        if c==34:
            val += H34[j]
        if c==35:
            val += H35[j]
        if c==36:
            val += H36[j]
        if c==37:
            val += H37[j]
        if c==38:
            val += H38[j]
        if c==39:
            val += H39[j]
        if c==40:
            val += H40[j]
        if c==41:
            val += H41[j]
        if c==42:
            val += H42[j]
        if c==43:
            val += H43[j]
        if c==44:
            val += H44[j]
        if c==45:
            val += H45[j]
        if c==46:
            val += H46[j]
        if c==47:
            val += H47[j]
        if c==48:
            val += H48[j]
        if c==49:
            val += H49[j]
        if c==50:
            val += H50[j]
        if c==51:
            val += H51[j]
        if c==52:
            val += H52[j]
        if c==53:
            val += H53[j]
        if c==54:
            val += H54[j]
        if c==55:
            val += H55[j]
        if c==56:
            val += H56[j]
        if c==57:
            val += H57[j]
        if c==58:
            val += H58[j]
        if c==59:
            val += H59[j]
        if c==60:
            val += H60[j]
        if c==61:
            val += H61[j]
        if c==62:
            val += H62[j]
        if c==63:
            val += H63[j]
        if c==64:
            val += H64[j]
        if c==65:
            val += H65[j]
        if c==66:
            val += H66[j]
        if c==67:
            val += H67[j]
        if c==68:
            val += H68[j]
        if c==69:
            val += H69[j]
        if c==70:
            val += H70[j]
        if c==71:
            val += H71[j]
        if c==72:
            val += H72[j]
        if c==73:
            val += H73[j]
        if c==74:
            val += H74[j]
        if c==75:
            val += H75[j]
        if c==76:
            val += H76[j]
        if c==77:
            val += H77[j]
        if c==78:
            val += H78[j]
        if c==79:
            val += H79[j]
        if c==80:
            val += H80[j]
        if c==81:
            val += H81[j]
        if c==82:
            val += H82[j]
        if c==83:
            val += H83[j]
        if c==84:
            val += H84[j]
        if c==85:
            val += H85[j]
        if c==86:
            val += H86[j]
        if c==87:
            val += H87[j]
        if c==88:
            val += H88[j]
        if c==89:
            val += H89[j]
        if c==90:
            val += H90[j]
        if c==91:
            val += H91[j]
        if c==92:
            val += H92[j]
        if c==93:
            val += H93[j]
        if c==94:
            val += H94[j]
        if c==95:
            val += H95[j]
        if c==96:
            val += H96[j]
        if c==97:
            val += H97[j]
        if c==98:
            val += H98[j]
        if c==99:
            val += H99[j]
        if c==100:
            val += H100[j]
        if c==101:
            val += H101[j]
        if c==102:
            val += H102[j]
        if c==103:
            val += H103[j]
        if c==104:
            val += H104[j]
        if c==105:
            val += H105[j]
        if c==106:
            val += H106[j]
        if c==107:
            val += H107[j]
        if c==108:
            val += H108[j]
        if c==109:
            val += H109[j]
        if c==110:
            val += H110[j]
        if c==111:
            val += H111[j]
        if c==112:
            val += H112[j]
        if c==113:
            val += H113[j]
        if c==114:
            val += H114[j]
        if c==115:
            val += H115[j]
        if c==116:
            val += H116[j]
        if c==117:
            val += H117[j]
        if c==118:
            val += H118[j]
        if c==119:
            val += H119[j]
        if c==120:
            val += H120[j]
        if c==121:
            val += H121[j]
    val /= len(H1)
    print(val)

# fig, ax = plt.subplots()
# xs = np.linspace(50, 450)
#
# print(len(H1))

# mean,std=norm.fit(H1)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.005')
#
# mean,std=norm.fit(H2)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.01')
#
# mean,std=norm.fit(H3)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.05')
#
# mean,std=norm.fit(H4)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.1')

# mean,std=norm.fit(H5)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.5')
#
# mean,std=norm.fit(H6)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=1')

# mean,std=norm.fit(H7)
# y = norm.pdf(xs, mean, std)
# plt.plot(xs, y,label='beta=0.9')




# density = gaussian_kde(H11)
# density._compute_covariance()
# # ax.plot(xs,density(xs),label='beta=0.01')
#
# density = gaussian_kde(H22)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.1')
#
# density = gaussian_kde(H33)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.2')
#
# density = gaussian_kde(H44)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.3')
#
# #
# density = gaussian_kde(H55)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.4')
# #
# density = gaussian_kde(H66)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.5')
#
# # density = gaussian_kde(H77)
# # density._compute_covariance()
# # ax.plot(xs,density(xs),label='beta=0.6')
# #
# # density = gaussian_kde(H88)
# # density._compute_covariance()
# # ax.plot(xs,density(xs),label='beta=0.7')
# #
# density = gaussian_kde(H99)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=0.8')
# #
# # density = gaussian_kde(H110)
# # density._compute_covariance()
# # ax.plot(xs,density(xs),label='beta=0.9')
#
# density = gaussian_kde(H121)
# density._compute_covariance()
# ax.plot(xs,density(xs),label='beta=1')



# plt.xlim([200, 400])
# plt.ylim([0, 0.04])
# ax.hist(H1, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.01')
# ax.hist(H2, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.1')
# ax.hist(H3, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.3')
# ax.hist(H4, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.6')
# ax.hist(H5, bins='auto', histtype=u'step', density=True, label = 'alpaha=0.9')
# ax.hist(H6, bins='auto', histtype=u'step', density=True, label = 'alpaha=1')

# ax.set_xlabel(r"Value of the utility function $f(\mathit{S^{B}}, y)$", fontsize=14)
# ax.set_ylabel(r"Probability distribution of $f(\mathit{S^{B}}, y)$", fontsize=14)
#
# plt.legend()
# plt.show()

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
