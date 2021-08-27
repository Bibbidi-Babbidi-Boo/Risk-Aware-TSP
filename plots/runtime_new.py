import numpy as np
import matplotlib.pyplot as plt

n_5 = np.array([1.299, 2.616, 0.5524, 9.6380, 2.1356, 2.6319])
n_10 = np.array([159.512, 41.6829, 92.4988, 7.4850, 275.7664098739624, 23.311601400375366])
n_15 = np.array([398.83418107032776, 714.6389183998108, 1225.872662782669, 590.4089200496674, 269.2295591831207, 46.3548150062561])
n_20 = np.array([4081.8803396224976, 1200.3423674106598, 1458.3801543712616, 667.0991151332855, 2153.5196719169617, 2378.8240904808044])
n_25 = np.array([3771.5979840755463, 3657.9790840148926, 1298.4267234802246, 2950.4932374954224, 1566.973874092102])
# n_5 = np.array([1.299, 2.616, 0.5524, 9.6380, 2.1356, 2.6319])

n_5_mean = np.mean(n_5)
n_15_mean = np.mean(n_15)
n_10_mean = np.mean(n_10)
n_20_mean = np.mean(n_20)
n_25_mean = np.mean(n_25)

n_5_std = np.std(n_5)
n_15_std = np.std(n_15)
n_10_std = np.std(n_10)
n_20_std = np.std(n_20)
n_25_std = np.std(n_25)

fig, ax = plt.subplots()

plt.plot(np.array([5, 10, 15, 20, 25]), np.array([n_5_mean, n_10_mean, n_15_mean, n_20_mean, n_25_mean])) 
plt.errorbar(np.array([5, 10, 15, 20, 25]), np.array([n_5_mean, n_10_mean, n_15_mean, n_20_mean, n_25_mean]), yerr=np.array([n_5_std, n_10_std, n_15_std, n_20_std, n_25_std]), fmt ='o', capsize=5)

ax.set_xlabel('Number of Nodes (N)')
ax.set_ylabel('Running Time (s)')

plt.show()
