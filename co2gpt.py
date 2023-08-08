import numpy as np
import numpy.linalg as la
import nlopt
import torch
import sys
import time
import math

def writeMatrix(Mat, filename):
    np.savetxt(filename, Mat)

def readMatrix(filename):
    return np.load(filename)

def norm(i, j):
    x1, y1 = i // lon, i % lon
    x2, y2 = j // lon, j % lon
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def exponential_cf(dist, sigma, l):
    return sigma**2 * math.exp(-dist / l)

def exponential_cf_l(dist, sigma, l):
    return sigma**2 * math.exp(-dist / l) * dist / l**2

def exponential_cf_sigma(dist, sigma, l):
    return sigma * 2.0 * math.exp(-dist / l)

def gen_cov_matrix(Mat, sigma, l, cfp, isR, prime=False):
    assert Mat.shape[0] == Mat.shape[1]
    for i in range(Mat.shape[0]):
        for j in range(i, Mat.shape[1]):
            if i == j:
                if not prime:
                    Mat[i, j] = sigma**2
                else:
                    Mat[i, j] = sigma * 2.0
            else:
                r = abs(i - j) if isR else norm(i, j)
                Mat[i, j] = cfp(r, sigma, l)
                Mat[j, i] = Mat[i, j]
                #print(i,", ",j)

def likelihood(x, grad):
    global count
    sigma_o, r_l, sigma_b, b_l = x
    print(x)

    H = data["H"]
    y = data["y"]
    x_prior = data["x_prior"]
    shape = data["shape"]
    obs_num = y.shape[0]
    grids_num = x_prior.shape[0]
    
    print("1")
    R_theta = np.zeros((obs_num, obs_num))
    gen_cov_matrix(R_theta, sigma_o, r_l, exponential_cf, True, False)

    print("2")
    tic = time.time()
    B_theta = np.zeros((grids_num, grids_num))
    gen_cov_matrix(B_theta, sigma_b, b_l, exponential_cf, False, False)
    toc = time.time()
    print("spend tim :",toc-tic)

    print("3")
    R_o = np.zeros((obs_num, obs_num))
    gen_cov_matrix(R_o, sigma_o, r_l, exponential_cf_sigma, True, True)

    print("4")
    R_l = np.zeros((obs_num, obs_num))
    gen_cov_matrix(R_l, sigma_o, r_l, exponential_cf_l, True, False)

    print("5")
    B_b = np.zeros((grids_num, grids_num))
    gen_cov_matrix(B_b, sigma_b, b_l, exponential_cf_sigma, False, True)

    print("6")
    B_l = np.zeros((grids_num, grids_num))
    gen_cov_matrix(B_l, sigma_b, b_l, exponential_cf_l, False, False)

    print("7")
    D_theta = R_theta + H @ B_theta @ H.T
    print("8")
    D_inv = la.inv(D_theta)
    print("9")
    alpha = D_inv @ y
    print("10")
    p1 = D_inv - alpha @ alpha.T

    print("11")
    if grad.size > 0:
        grad[0] = 0.5 * np.trace(p1 @ R_o)
        print("12")
        grad[1] = 0.5 * np.trace(p1 @ R_l)
        print("13")
        grad[2] = 0.5 * np.trace(p1 @ H @ B_b @ H.T)
        grad[3] = 0.5 * np.trace(p1 @ H @ B_l @ H.T)

    print("14")
    y_Hxp = y - H @ x_prior
    (sign, logabsdet) = la.slogdet(D_theta)
    ret1 = sign * logabsdet
    ret2 = y_Hxp.T @ D_inv @ y_Hxp
    ret = ret1 + ret2
    print(ret)
    ret.astype(np.float64)
    count = count + 1
    return ret[0,0]


y = readMatrix("./data/hhsd/y.npy")
H = readMatrix("./data/hhsd/H.npy")
x_prior = readMatrix("./data/hhsd/x_prior.npy")
flux = readMatrix("./data/hhsd/flux.npy")
shape = readMatrix("./data/hhsd/shape.npy")
print("read data from file")
count = 0

lon= shape[0]

data = {"H": H, "y": y, "x_prior": x_prior, "shape": shape}

x0 = np.array([10.0, 20.0, 10.0, 20.0])
opt = nlopt.opt(nlopt.GN_ISRES,x0.size)
opt.set_min_objective(likelihood)
lb = [0.0, 0.0, 0.0, 0.0]
ub = [20.0, 40.0, 20.0, 40.0]
opt.set_lower_bounds(lb)
opt.set_upper_bounds(ub)
opt.set_maxtime(60)
xopt = opt.optimize(x0)
opt_val = opt.last_optimum_value()
res = opt.last_optimize_result()

if res>0:
    print("found minimum at f({}, {}, {}, {}) = {:.10f}".format(xopt[0], xopt[1], xopt[2], xopt[3], opt_val))
    print("optim times : ",count)
else:
    print("nlopt failed!")
    print(res)
sys.exit()
sigma_o, r_l, sigma_b, b_l = xopt
obs_num = y.shape[0]
grids_num = x_prior.shape[0]
R_theta = np.zeros((obs_num, obs_num))
gen_cov_matrix(R_theta, sigma_o, r_l, exponential_cf, True, False)

B_theta = np.zeros((grids_num, grids_num))
gen_cov_matrix(B_theta, sigma_b, b_l, exponential_cf, False, False)

D_theta = R_theta + H @ B_theta @ H.T
D_inv = la.inv(D_theta)

posterior = x_prior + B_theta @ H.T @ D_inv @ (y - H @ x_prior)
diff = posterior - flux
se = diff.T @ diff
rmse = np.sqrt(se / grids_num)

writeMatrix(posterior, "posterior.txt")
writeMatrix(x_prior, "x_prior.txt")
writeMatrix(flux, "flux.txt")
print("posterior sum =", posterior.sum())
print("prior sum =", x_prior.sum())
print("true flux sum =", flux.sum())
print("rmse =", rmse)

