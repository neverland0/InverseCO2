import numpy as np
import nlopt
np.set_printoptions(threshold = np.inf)
np.set_printoptions(suppress=True)

file_flux = 'flux.csv'

F = np.genfromtxt(file_flux,delimiter=",",skip_header=1,usecols=range(1,121))
P = np.genfromtxt(file_flux,delimiter=",")
#print(F)

out = np.zeros((50,30))
for i in range(250):
    for j in range(120):
        x = i // 5
        y = j // 4
        out[x][y] += F[i][j]

out = out/20
for i in range(50):
    for j in range(30):
        if out[i][j]:
            out[i][j] = out[i][j] - 0.2 * out[i][j] * np.random.normal(1,0.5)

for i in range(250):
    for j in range(120):
        x = i // 5
        y = j // 4
        P[i+1][j+1] = out[x][y]

print(P)
np.savetxt('prior_new.csv',P,fmt='%f',delimiter=',')
