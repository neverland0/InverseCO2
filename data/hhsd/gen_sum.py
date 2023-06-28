import netCDF4 as nc
import sys
import numpy as np
files = sys.argv[1:]
lon = np.linspace(112.005,114.495,250)
lat = np.linspace(34.005,35.195,120)
print(lon.size)
print(lat.size)
#shape of H is (lat,lon)
shape = (lat.size,lon.size)
H_sum = np.zeros(shape)
print(H_sum)

for f in files:
    file = nc.Dataset(f)
    time = file.variables['time']
    time_size = time.size
    for t in np.arange(time_size):
        Hi = file.variables['foot'][:][:][t]
        H_sum += Hi

print(H_sum)
print(np.sum(H_sum))
