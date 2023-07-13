import numpy as np
np.set_printoptions(suppress=True)


sigma_o = 5
r_l = 2
sigma_b = 5
b_l = 10

s = np.load('shape.npy')
lon = s[0]
lat = s[1]

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
    cov = p[grid1][0] * p[grid2][0] * np.exp(-norm/10)
    return cov
def cal_cov2(grid1,grid2):
    norm = get_norm(grid1,grid2)
    cov = sigma_b * sigma_b * np.exp(-norm/b_l)
    return cov
def cal_cov_s(grid1,grid2):
    norm = np.abs(grid1-grid2)
    cov = np.square(sigma_o) * np.exp(-2*np.square(np.sin(norm/2))/np.square(r_l))
    return cov
def cal_cov_r2(grid1,grid2):
    norm = np.abs(grid1-grid2)
    cov = np.square(sigma_o) * np.exp(-norm/r_l)
    return cov


y = np.load('y.npy')
H = np.load('H.npy')
x_prior = np.load('x_prior.npy')
obs_num = H.shape[0]
grids = H.shape[1]
print("obs_num,grids = ",obs_num," ",grids)
B = np.zeros((grids,grids))
R = np.zeros((obs_num,obs_num))
flux = np.load('flux.npy')

#for i in range(obs_num):
#    R[i][i] = 25
    #R[i][i] = sigma_o * sigma_o
for i in range(obs_num):
    for j in range(obs_num):
        R[i][j] = cal_cov_r2(i,j)
for i in range(grids):
    for j in range(grids):
        #B[i][j] = cal_cov(i,j,x_prior)
        B[i][j] = cal_cov2(i,j)
print("R:")
print(R)
print("B:")
print(B)
BHt = np.dot(B,H.T)
HBHt = np.dot(np.dot(H,B),H.T)
Hx_prior = np.dot(H,x_prior)
print('BHT:',BHt.shape,'\n',BHt)
print('HBHT:',HBHt.shape,'\n',HBHt)
print('Hx_prior:',Hx_prior.shape,'\n',Hx_prior)

p_1 = HBHt+R
p_2 = np.linalg.inv(p_1)
p_3 = y-Hx_prior
p_4 = np.dot(BHt,p_2)
p_5 = np.dot(p_4,p_3)

print('p_3:',p_3.shape,'\n',p_3)
print('p_5:',p_5.shape,'\n',p_5)
print('x_prior',x_prior.shape,'\n',x_prior)

x_posterior = x_prior + p_5

print(x_prior.sum())
print("posterior = ",x_posterior.sum())

n = flux.shape[0]
x_diff = x_posterior - flux
se = np.vdot(x_diff,x_diff)
mse = se / n
rmse = np.sqrt(mse)
x_hat = x_prior
x_diff2 = x_prior - flux
sst = np.vdot(x_diff2,x_diff2)
r2 = 1- se / sst

print("rmse = ",rmse," r2 = ",r2)

print("x_prior")
#print(x_prior)
print("flux")
#print(flux)
print("x_posterior")
print(x_posterior)
print("flux:")
print(flux.sum())
