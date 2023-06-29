import netCDF4 as nc
import sys
import numpy as np
np.set_printoptions(threshold = np.inf)
files = sys.argv[1:]
file_flux = '../flux.csv'
file_prior = '../prior.csv'
F = np.genfromtxt(file_flux,delimiter=",",usecols=range(1,121),skip_header=1)
print(F.shape)
#print(F)
moments = len(files)
obs = []
H = []
lon = np.linspace(112.005,114.495,250)
lat = np.linspace(34.005,35.195,120)
print(lon.size)
print(lat.size)
#shape of H is (lon,lat)
shape = (lon.size,lat.size)
#H_sum = np.zeros(shape)
#print(H_sum)

for f in files:
    file = nc.Dataset(f)
    time = file.variables['time']
    time_size = time.size
    H_p = np.zeros(shape)
    for t in np.arange(time_size):
        Hi = file.variables['foot'][:][:][t].T
        #H_sum += Hi
        H_p += Hi
    H.append(H_p.flatten('C'))
    obs_p = np.vdot(H_p,F)
    obs.append(obs_p)
print("obs:",obs)
print("obs num:",len(obs))

H = np.vstack(H)
def get_coor(gird):
    x = grid // lon.size
    y = grid % lon.size
    return x,y
def get_norm(grid1,grid2):
    x1,y1 = get_coor(grid1)
    x2,y2 = get_coor(grid2)
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))
def cal_cov(grid1,grid2,f):
    cov = f[grid1] * f[grid2] * np.exp(-get_norm(grid1,grid2)/20)
#print(H)
#print(H[400])
y = np.array(obs).flatten().reshape(-1,1)
x_prior = F.flatten().reshape(-1,1)
grids = lon.size * lat.size
B = np.zeros((grids,grids))
R = np.zeros((moments,moments))
for i in range(moments):
    R[i][i] = 25
for i in range(grids,grids):
    for j in range(grids,grids):
        B[i][j] = cal_cov(i,j,F.flatten())
np.save('y.npy',y)
np.save('H.npy',H)
np.save('x_prior.npy',x_prior)
np.save('R.npy',R)
np.save('B.npy',B)
np.savetxt('y.txt',y,delimiter=',')
np.savetxt('H.txt',H,delimiter=',')
np.savetxt('x_prior.txt',x_prior,delimiter=',')
np.savetxt('R.txt',R,delimiter=',')
np.savetxt('B.txt',B,delimiter=',')
