import matplotlib.pyplot as plt
import numpy as np

values = np.loadtxt("taylor_dir.txt", delimiter=',')

epsilon = [i[0] for i in values]
epsilon = np.asarray(epsilon)

omegas = [i[1] for i in values]
omega_FD = np.abs(omegas - omegas[0])
print(omega_FD)
omega_prime = [i[2] for i in values]

omega_AD = epsilon*abs(omega_prime[0])
print(omega_AD)
xs = epsilon**2
ys = np.abs(omega_AD - omega_FD)

xs = xs[1:]
ys = ys[1:]

# print(xs)
# print(ys)

import matplotlib 
font = {'size'   : 13}

matplotlib.rc('font', **font)

plt.figure(figsize=(8, 4))
plt.plot([xs[0], xs[-1]], [ys[0], ys[-1]], color='0.5', linestyle='--')
plt.plot(xs, ys, color='black')
plt.scatter(xs,ys,color='r')
plt.xlabel('$\epsilon^2$')
plt.ylabel('$|\delta_{FD} - \delta_{AD}|$')
plt.tight_layout()
plt.savefig("taylor2Dcircle.pdf")
plt.show()