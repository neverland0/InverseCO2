import netCDF4 as nc
import sys
import numpy as np
np.set_printoptions(threshold = np.inf)
files = sys.argv[1:]
file_flux = '../flux.csv'
file_prior = '../prior.csv'
obs_num = len(files)
print(obs_num," .nc file has been read")
# zhengzhou city area
#start_point = (28,73)
#end_point = (99,222)

# zhengzhou centra area
start_point = (57,124)
end_point = (77,169)

# small area
#start_point = (88,157)
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
        #print(H_p)
        #print(H_p.flatten('C'))
    H.append(H_p.flatten('C'))
    obs_p = np.vdot(H_p,F) + np.random.normal(0,0.5)
    obs.append(obs_p)

H = np.vstack(H)

y = np.array(obs).flatten().reshape(-1,1)
x_prior = P.flatten().reshape(-1,1)
flux = F.flatten().reshape(-1,1)
#grids = lon * lat
np.save('y.npy',y)
np.save('H.npy',H)
np.save('x_prior.npy',x_prior)
np.save('shape.npy',shape)
np.save('flux.npy',flux)
print("writting to file.npy")
np.savetxt('y.txt',y,delimiter=' ')
np.savetxt('H.txt',H,delimiter=' ')
np.savetxt('x_prior.txt',x_prior,delimiter=' ')
np.savetxt('shape.txt',shape,delimiter=' ')
np.savetxt('flux.txt',flux,delimiter=' ')
print("writting to file.txt")
