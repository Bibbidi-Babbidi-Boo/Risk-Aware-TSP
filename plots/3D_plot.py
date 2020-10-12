from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

## needs to be square array
Z = np.array([[349.4700598073035, 407.318377678, 420.507823969, 495.531435498, 654.3848727743666, 653.034177783], [426.805583681806, 430.088782509, 465.057533485, 533.194438519, 553.377150052, 713.7584958912032], [399.52059890261523, 412.642391849, 444.400234972, 657.3965264983747, 562.110940889, 687.8719794083187], [413.8545383924215, 446.875539677, 487.741085204, 543.710573968, 603.999669051, 761.165921381081], [431.28651934650065, 423.86386882, 486.191100074, 517.889452065, 622.201343508, 742.3712060466428], [431.28651934650065, 423.86386882, 486.191100074, 517.889452065, 622.201343508, 742.3712060466428]])
# print(z.shape)
X = np.array([[0, 0.1, 0.3, 0.5, 0.7, 1], [0, 0.1, 0.3, 0.5, 0.7, 1], [0, 0.1, 0.3, 0.5, 0.7, 1], [0, 0.1, 0.3, 0.5, 0.7, 1], [0, 0.1, 0.3, 0.5, 0.7, 1], [0, 0.1, 0.3, 0.5, 0.7, 1]])
Y = np.array([[0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
# x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))
# print(x.shape, y.shape, z.shape)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel('beta')
ax.set_ylabel('alpha')
ax.set_zlabel('reward')

plt.show()


#
# # show hight map in 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('beta')
# ax.set_ylabel('alpha')
# ax.set_zlabel('reward')
# ax.plot_surface(x, y, z)
# plt.title('reward height map')
# plt.show()
#
# show hight map in 2d
plt.figure()
plt.title('reward heat map')
ax.set_xlabel('beta')
ax.set_ylabel('alpha')
ax.set_zlabel('reward')
p = plt.imshow(Z)
plt.colorbar(p)
plt.show()
