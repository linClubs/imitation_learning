from random import seed
import numpy as np
seed(0)
a = np.random.randn(3, 4)
print(a)
b = np.random.randn(4, 5)

c = np.random.randint(1, 10, [2,3])

d = np.random.random([4,2])
print(b)
print(c)
print(a@d)
