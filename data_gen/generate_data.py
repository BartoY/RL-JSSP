import numpy as np
from uniform_instance_gen import uni_instance_gen

j = 10
m = 10
l = 0.01
h = 1.00
batch_size = 512
seed = 200

np.random.seed(seed)

data = np.array([uni_instance_gen(n_j=j, n_m=m, low=l, high=h) for _ in range(batch_size)])
print(data.shape)
np.save('generatedData{}_{}_Seed{}_bsz{}.npy'.format(j, m, seed,batch_size), data)