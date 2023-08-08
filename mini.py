import torch
import math
import torch.optim as optim
import numpy as np

def writeMatrix(Mat, filename):
    np.savetxt(filename, Mat, delimiter=' ')
    np.save(filename, Mat)

def readMatrix(filename):
    return np.load(filename)

def norm(i, j):
    x1, y1 = i // lon, i % lon
    x2, y2 = j // lon, j % lon
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


# 定义自定义函数，这里使用简单的三次函数作为例子
def custom_function(l):
    sigma_o = l[0]
    r_l = l[1]
    sigma_b = l[2]
    b_l = l[3]
    print("x1=",sigma_o,"x2=",r_l,"x3=",sigma_b,"x4=",b_l)

    R = torch.zeros((obs_num,obs_num))
    print("12")
    B = torch.zeros((grids_num,grids_num))
    print("13")

    for i in range(obs_num):
        for j in range(i, obs_num):
            if i == j:
                R[i, j] = sigma_o**2
                #print("r ",i,',',j)
            else:
                r = abs(i - j)
                R[i, j] = sigma_o**2 * np.exp(-r / r_l)
                R[j, i] = R[i, j]
                print("r ",i,',',j)
    print("13b")
    for i in range(grids_num):
        for j in range(i, grids_num):
            if i == j:
                B[i, j] = sigma_b**2
                #print("b ",i,',',j)
            else:
                r = norm(i, j)
                B[i, j] = sigma_b**2 * np.exp(-r / b_l)
                B[j, i] = B[i, j]
                print("b ",i,',',j)
    print("13c")
    #R = torch.from_numpy(R).to(torch.float32)
    #B = torch.from_numpy(B).to(torch.float32)
    #B = B.to(torch.float32)
    #B = B.to("cuda:0")
    print("14")
    D = torch.zeros((obs_num,obs_num))
    print("15")
    
    print("R is cuda" if R.is_cuda else "R is cpu")
    print("B is cuda" if B.is_cuda else "B is cpu")
    print("D is cuda" if D.is_cuda else "D is cpu")
    print("16")
    D = R + H @ B @ H.T
    print("17")
    y_Hx = y - H @ x_prior
    print("18")
    D_inv = torch.inverse(D)
    print("19")
    ret = torch.logdet(D) + y_Hx.T @ D_inv @ y_Hx
    print("20")
    return ret


y = readMatrix("./data/hhsd/y.npy")
H = readMatrix("./data/hhsd/H.npy")
x_prior = readMatrix("./data/hhsd/x_prior.npy")
flux = readMatrix("./data/hhsd/flux.npy")
shape = readMatrix("./data/hhsd/shape.npy")
print("read data from file")
obs_num = y.shape[0]
print("1obs_num",obs_num)
grids_num = x_prior.shape[0]
print("2grids_num",grids_num)
lon = shape[0]
print("3")
y = torch.tensor(y).to(torch.float32)
print("y is cuda" if y.is_cuda else "y is cpu")
print("4")
H = torch.tensor(H).to(torch.float32)
print("H is cuda" if H.is_cuda else "H is cpu")
print("5")
x_prior = torch.tensor(x_prior).to(torch.float32)
print("6")
flux = torch.tensor(flux).to(torch.float32)

# 初始化参数，随机选择起始点，并将它们移动到CUDA设备上
sigma_o = torch.tensor([10.0], requires_grad=False)
r_l = torch.tensor([20.0], requires_grad=False)
sigma_b = torch.tensor([10.0], requires_grad=False)
b_l = torch.tensor([20.0], requires_grad=False)


l = [sigma_o,r_l,sigma_b,b_l]
# 定义优化器，这里使用随机梯度下降（SGD）优化器
optimizer = optim.Adam(l, lr=0.1)

# 迭代优化过程
#with torch.cuda.device('cuda:0'):  # 将计算迁移到CUDA设备上
for _ in range(100):
    # 计算函数值

    print("7")
    loss = custom_function(l)
    print("8")
    
    # 反向传播计算梯度
    optimizer.zero_grad()
    print("9")
    loss.backward()
    
    print("10")
    # 更新参数
    optimizer.step()

    print("11")
# 打印最终结果
print("最小化的 sigma_o:", l[0].item())
print("最小化的 r_l:", l[1].item())
print("最小化的 sigma_b:", l[2].item())
print("最小化的 b_l:", l[2].item())
print("最小化的函数值:", custom_function(l).item())

