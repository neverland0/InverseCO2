import netCDF4 as nc
import sys
import numpy as np
np.set_printoptions(threshold = np.inf)
files = sys.argv[1:]
file_flux = '../flux.csv'
file_prior = '../prior.csv'
obs_num = len(files)
print(obs_num," .nc file has been read")
#F = np.genfromtxt(file_flux,delimiter=",",usecols=range(1,121),skip_header=1)
F = np.genfromtxt(file_flux,delimiter=",",usecols=range(87,97),skip_header=156,skip_footer=75)
P = np.genfromtxt(file_prior,delimiter=",",usecols=range(87,97),skip_header=156,skip_footer=75)
#lon = np.linspace(112.005,114.495,250)
#lat = np.linspace(34.005,35.195,120)
lon = F.shape[0]
lat = F.shape[1]
shape = (F.shape)
print("target area has the shape of ",shape)
print(F)
print(P)
print(F.sum())
print(P.sum())
#shape of H is (lon,lat)
#shape = (lon.size,lat.size)
H_shape = (obs_num,lon*lat)
H = []
obs = []
for f in files:
    file = nc.Dataset(f)
    time = file.variables['time']
    time_size = time.size
    H_p = np.zeros(shape)
    for t in np.arange(time_size):
        Hi = file.variables['foot'][t,86:96,155:175].T
        #H_sum += Hi
        H_p += Hi
    H.append(H_p.flatten('C'))
    obs_p = np.vdot(H_p,F) + np.random.normal(0,0.5)
    obs.append(obs_p)
print("obs:",obs)
print("obs num:",len(obs))

H = np.vstack(H)
print('H shape=',H.shape)
#print(H)
def get_coor(grid):
    x = grid // lon
    y = grid % lon
    return x,y
def get_norm(grid1,grid2):
    x1,y1 = get_coor(grid1)
    x2,y2 = get_coor(grid2)
    ret = np.sqrt(np.square(x1-x2)+np.square(y1-y2))
    return ret
def cal_cov(grid1,grid2,p):
    norm = get_norm(grid1,grid2)
    cov = p[grid1] * p[grid2] * np.exp(-norm/10)
    return cov
#print(H)
#print(H[400])
y = np.array(obs).flatten().reshape(-1,1)
x_prior = P.flatten().reshape(-1,1)
grids = lon * lat
print("grids=",grids)
B = np.zeros((grids,grids))
R = np.zeros((obs_num,obs_num))
for i in range(obs_num):
    R[i][i] = 25
for i in range(grids):
    for j in range(grids):
        B[i][j] = cal_cov(i,j,P.flatten())
print("writting to file.npy")
np.save('y.npy',y)
np.save('H.npy',H)
np.save('x_prior.npy',x_prior)
np.save('R.npy',R)
np.save('B.npy',B)
print("writting to file.txt")
np.savetxt('y.txt',y,delimiter=',')
np.savetxt('H.txt',H,delimiter=',')
np.savetxt('x_prior.txt',x_prior,delimiter=',')
np.savetxt('R.txt',R,delimiter=',')
np.savetxt('B.txt',B,delimiter=',')
