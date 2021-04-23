import matplotlib.pyplot as plt
import numpy as np

bast_solution = np.load('./save/FashionMNIST/test1/best_solution.npy', allow_pickle=True)
solution = np.load('./save/cifar-10/test1/solution0.npy')
param = np.load('./save/cifar-100/test4/param0.npy')
learn_curve = np.load('./save/FashionMNIST/test1/learn_curve.npy', allow_pickle=True)

for i in range(5):
    fps = []
    for p in param:
        fps.append(p[0][i])
    plt.plot(fps, label='kernel stride {}'.format(i))
# plt.plot(bast_solution)
# plt.plot(solution)

# plt.xlim(-1, 40)
# plt.ylim(20, 95)
plt.xlabel('iterations')
plt.ylabel('range')
plt.legend()
plt.show()
