import netCDF4 as nc
import sys
files = sys.argv[1:]
first = files[0]
first_file = nc.Dataset(first)
print(first_file)
print(first_file.variables.keys())
print('Hii:')
print(first_file.variables['foot'][:][:][0])
#print(first_file.variables['lon'][:])
#print(first_file.variables['lat'][:])
print('shape of Hii =',first_file.variables['foot'][:][:][0].shape)
print('file_name',"\t\t\t\t\t",'time_dim')
for f in files:
    file = nc.Dataset(f)
    lon = file.variables['lon']
    lat = file.variables['lat']
    time = file.variables['time']
    foot = file.variables['foot']
    time = nc.num2date(time[:],time.units)
    print(f,"\t",time.size)
    print(time)
#for i in range(120):
#    print(foot[1][i])
a = np.arange(3*4).reshape(3,4)
b = np.arange(3*4).reshape(3,4)
c=a+b
print(a)
print(b)
print(c)
c_f=c.flatten(order='C')
print(c_f)
d=np.zeros((2,12))
d[0]=c_f
d[1]=c_f
d=np.r_[d,np.expand_dims(c_f,axis=0)]
print(d)
print(d)
