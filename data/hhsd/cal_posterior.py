import numpy as np

y = np.load('y.npy')
H = np.load('H.npy')
x_prior = np.load('x_prior.npy')
R = np.load('R.npy')
B = np.load('B.npy')

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

print(x_posterior.sum())
