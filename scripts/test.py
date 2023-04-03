import numpy as np

emp = np.empty((0, 3))

rand = np.arange(15).reshape(5, 3)
print(rand)

for i in range(rand.shape[0]):
    emp = np.concatenate((emp, rand[i, :].reshape(1, 3)), axis=0)
    # print(rand[i, :].shape)

print(emp)