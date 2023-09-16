import netCDF4 as nc
import sys
import numpy as np
np.set_printoptions(threshold = np.inf)
files = sys.argv[1:]
file_flux = '../flux.csv'
file_prior = '../prior.csv'
obs_num = len(files)
groups = obs_num // 744
if(groups > 1):
    obs_num = obs_num - 744
    verify = files[obs_num:]
    files = files[:obs_num]
print(obs_num," .nc file has been read")
# zhengzhou city area 5000 grids
#start_point = (48,123)
#end_point = (99,222)

# zhengzhou city area 1000 grids
#start_point = (50,110)
#end_point = (74,149)
# zhengzhou centra area 210 grids
#start_point = (58,127)
#end_point = (77,161)
# zhengzhou gaoxin area 2100 grids
start_point = (60,116)
end_point = (89,185)

# small area
#start_point = (94,172)
#end_point = (97,176)
#F = np.genfromtxt(file_flux,delimiter=",",usecols=range(1,121),skip_header=1)
#F1 = np.genfromtxt(file_flux,delimiter=",",usecols=range(87,97),skip_header=156,skip_footer=75)
#P1 = np.genfromtxt(file_prior,delimiter=",",usecols=range(87,97),skip_header=156,skip_footer=75)
x1 = start_point[0]
y1 = start_point[1]
x2 = end_point[0]
y2 = end_point[1]
F = np.genfromtxt(file_flux,delimiter=",",usecols=range(x1-1,x2),skip_header=y1-1,skip_footer=251-y2)
P = np.genfromtxt(file_prior,delimiter=",",usecols=range(x1-1,x2),skip_header=y1-1,skip_footer=251-y2)
#lon = np.linspace(112.005,114.495,250)
#lat = np.linspace(34.005,35.195,120)
shape = (F.shape)
lon = shape[0]
lat = shape[1]
print("target area has the shape of ",shape)
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
        Hi = file.variables['foot'][t,x1-2:x2-1,y1-2:y2-1].T
        H_p += Hi
    H.append(H_p.flatten('C'))
    obs_p = np.vdot(H_p,F) + np.random.normal(0,0.5)
    obs.append(obs_p)

H = np.vstack(H)

H_v = []
obs_v = []
if(groups > 1):
    for v in verify:
        ver = nc.Dataset(v)
        time1 = ver.variables['time']
        time_size1 = time1.size
        H_p1 = np.zeros(shape)
        for t1 in np.arange(time_size1):
            Hi1 = ver.variables['foot'][t1,x1-2:x2-1,y1-2:y2-1].T
            H_p1 += Hi1
        H_v.append(H_p1.flatten('C'))
        obs_p1 = np.vdot(H_p1,F)
        obs_v.append(obs_p1)
    H_v = np.vstack(H_v)

y = np.array(obs).flatten().reshape(-1,1)
y_v = np.array(obs_v).flatten().reshape(-1,1)
x_prior = P.flatten().reshape(-1,1)
flux = F.flatten().reshape(-1,1)
#grids = lon * lat
np.save('y.npy',y)
np.save('H.npy',H)
np.save('x_prior.npy',x_prior)
np.save('shape.npy',shape)
np.save('flux.npy',flux)
if(groups > 1):
    np.save('H_v.npy',H_v)
    np.savetxt('H_v.txt',H_v,delimiter=' ')
    np.save('obs_v.npy',obs_v)
    np.savetxt('obs_v.txt',obs_v,delimiter=' ')
print("writting to file.npy")
np.savetxt('y.txt',y,delimiter=' ')
np.savetxt('H.txt',H,delimiter=' ')
np.savetxt('x_prior.txt',x_prior,delimiter=' ')
np.savetxt('shape.txt',shape,delimiter=' ')
np.savetxt('flux.txt',flux,delimiter=' ')
print("writting to file.txt")
