# %%
import numpy as np
import struct
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import scale
import cv2 as cv
from math import floor, ceil
import math
import time

# %%
def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

with open('../scenes/sample.dat', 'rb') as f:
    fileContent = f.read()
    width, height, spp, SL = struct.unpack("LLLL", fileContent[:32])
    samples = struct.unpack("f" * ((len(fileContent) - 32) // 4), fileContent[32:])
    raw = np.array(samples)
    samples = np.reshape(raw, (width, height, spp, SL))
print(width, height, spp, SL)

# %%
  # render color
raw = np.zeros([width, height, 3])
raw[:,:,2] = samples[:,:,0:8,2].mean(-1)
raw[:,:,1] = samples[:,:,0:8,3].mean(-1)
raw[:,:,0] = samples[:,:,0:8,4].mean(-1)

cv.imwrite('../testres/raw.jpg', raw * 255.0)

# render position

positions = np.zeros([width, height, 3])
positions[:,:,2] = normalize(samples[:,:,0:8,5].mean(-1))
positions[:,:,1] = normalize(samples[:,:,0:8,6].mean(-1))
positions[:,:,0] = normalize(samples[:,:,0:8,7].mean(-1))

cv.imwrite('../testres/position.jpg', positions * 255.0)

# render normal

normals = np.zeros([width, height, 3])
normals[:,:,2] = normalize(samples[:,:,0:8,8].mean(-1))
normals[:,:,1] = normalize(samples[:,:,0:8,9].mean(-1))
normals[:,:,0] = normalize(samples[:,:,0:8,10].mean(-1))

cv.imwrite('../testres/normal.jpg', normals * 255.0)

# render texture

textures = np.zeros([width, height, 3])
textures[:,:,2] = samples[:,:,0:8,11].mean(-1)
textures[:,:,1] = samples[:,:,0:8,12].mean(-1)
textures[:,:,0] = samples[:,:,0:8,13].mean(-1)

cv.imwrite('../testres/texture.jpg', textures * 255.0)
# print(samples[10,5,4,:])    

# %%

def getNeighbourIndex(x, y, b, M):
    P = np.repeat(np.array((x, y)).reshape(1,2),spp,axis=0)
    P = np.concatenate((P, np.arange(0,spp).reshape(spp,1)), axis=1)
    
    indexN = np.random.multivariate_normal(mean=[x, y], cov=(b/4) * np.identity(2), size=M-spp)
    indexN = np.clip(indexN, 0, width - 1) # TODO : column wise clip
    indexN = np.around(indexN)
    randSampleInPixel = np.random.randint(7, size=(M-spp, 1))
    indexN = np.concatenate((indexN,randSampleInPixel), axis = 1)
    indexN = np.concatenate((P, indexN.astype(int)), axis=0)
    return P, indexN

def preprocess(x, y, b, M):
    # Sample neighbors based on gaussian distribution
    P, indexN = getNeighbourIndex(x, y, b, M)
    sampleMapper = lambda v: samples[(v[0]), (v[1]), (v[2]),:]
    P = np.apply_along_axis(sampleMapper, 1, P)
    N = np.apply_along_axis(sampleMapper, 1, indexN)

    mpf = np.repeat(np.mean(P, axis=0).reshape(1, P.shape[1]), N.shape[0], axis=0)
    sigma_pf = np.repeat(np.std(P, axis=0).reshape(1, P.shape[1]), N.shape[0], axis=0)
    
    cond1 = np.absolute(N - mpf) > 30 * sigma_pf
    cond2 = np.absolute(N - mpf) > 0.1
    cond3 = sigma_pf > 0.1
    
    res = cond1 & (cond2 | cond3)
    
    mask = ~(np.max(res.astype(int), axis=1).astype(bool))
    mask[0:28] = True
    
    N = N[mask]
    indexN = indexN[mask]
    # Standardize neighbors
    # N = np.concatenate((P, N), axis=0)
    
    # N = (N - np.mean(N, axis=0)) / np.std(N, axis=0)
    setColor = N[:, 2:5]
    N = scale(N)
    
    return N, P, indexN, setColor

# %%
def sumD(Dxx):
    return Dxx[0] + Dxx[1] + Dxx[2]

def mutual_info(x, y):
    c_xy = np.histogram2d(x, y)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def computeFeatureWeights(t, N):
    epsilon = 1e-10

    # cn = [2, 3, 4] # R, G, B
    # rn = [0, 1, 23, 24] # X, Y, U, V
    # pn = [0, 1] # X, Y
    
    # FEATURECOUNT = 12
    # # f = N[:,2:14]
    # f = list(range(2, 14))
    
    # Dcr = [0.0, 0.0, 0.0] # Deps between color and random
    # Dcp = [0.0, 0.0, 0.0] # Deps between color and position
    # Dcf = [0.0, 0.0, 0.0] # Deps between color and feature
    # Wcr = [0.0, 0.0, 0.0] # 
    # alpha = [0.0, 0.0, 0.0]
    
    # Dfc = np.zeros(FEATURECOUNT)
    # for k in range(3):
    #     for j in range(4):
    #         Dcr[k] += mutual_info(N[:,cn[k]], N[:,rn[j]])
    #     for l in range(2):
    #         Dcp[k] += mutual_info(N[:,cn[k]], N[:,pn[l]])
    #     for l in range(12):
    #         tmp = mutual_info(N[:,cn[k]],N[:,f[l]])
    #         Dcf[k] += tmp
    #         Dfc[l] += tmp

    #     Wcr[k] = Dcr[k] / (Dcr[k] + Dcp[k] + epsilon)
    #     # alpha[k] = max(1 - 2 * (1 + 0.1 * t) * Wcr[k], 0)
    #     alpha[k] = 1 - Wcr[k]

    # Dfr = np.zeros(FEATURECOUNT)
    # Dfp = np.zeros(FEATURECOUNT)
    # Wfr = np.zeros(FEATURECOUNT)
    # Wcf = np.zeros(FEATURECOUNT)
    # beta = np.zeros(FEATURECOUNT)
    
    # for k in range(FEATURECOUNT):
    #     for l in range(4):
    #         Dfr[k] += mutual_info(N[:,f[k]], N[:,rn[l]])
    #     for l in range(2):
    #         Dfp[k] += mutual_info(N[:,f[k]], N[:,pn[l]])
        
    #     Wfr[k] = Dfr[k] / (Dfr[k] + Dfp[k] + epsilon)

    #     Wcf[k] = Dfc[k] / (sumD(Dcr) + sumD(Dcp) + sumD(Dcf))
    #     # beta[k] = max(1 - (1 + 0.1 * t) * Wfr[k], 0)
    #     beta[k] = Wcf[k] * (1 - Wfr[k])

    # return alpha, beta, Wcr

    c_bar = N[:,2:5]
    r_bar = np.concatenate((N[:,0:2],N[:,23:25]), axis=1)
    p_bar = N[:,0:2]
    f_bar = N[:,2:14]
    
    Dcr_buf = np.indices((3,4)).transpose(1,2,0).reshape(12,2)
    Dcr_buf = np.apply_along_axis(lambda v: mutual_info(c_bar[:,v[0]], r_bar[:,v[1]]), 1, Dcr_buf).reshape(3,4)
    Dcr_buf = np.sum(Dcr_buf, axis=1)
    
    Dcp_buf = np.indices((3,2)).transpose(1,2,0).reshape(6,2)
    Dcp_buf = np.apply_along_axis(lambda v: mutual_info(c_bar[:,v[0]], p_bar[:,v[1]]), 1, Dcp_buf).reshape(3,2)
    Dcp_buf = np.sum(Dcp_buf, axis=1)
    
    Dcf_buf = np.indices((3,12)).transpose(1,2,0).reshape(36,2)
    Dcf_buf = np.apply_along_axis(lambda v: mutual_info(c_bar[:,v[0]], f_bar[:,v[1]]), 1, Dcf_buf).reshape(3,12)
    
    Dfc_buf = np.sum(Dcf_buf, axis=0)
    Dcf_buf = np.sum(Dcf_buf, axis=1)
    

    Wcr_buf = Dcr_buf / (Dcr_buf + Dcp_buf + epsilon)
    # alpha_buf = np.maximum(1 - 2 * (1 + 0.1 * t) * Wcr_buf, np.zeros(3))
    alpha_buf = 1 - Wcr_buf

    Dfr_buf = np.indices((12,4)).transpose(1,2,0).reshape(48,2)
    Dfr_buf = np.apply_along_axis(lambda v: mutual_info(f_bar[:,v[0]], r_bar[:,v[1]]), 1, Dfr_buf).reshape(12,4)
    Dfr_buf = np.sum(Dfr_buf, axis=1)

    Dfp_buf = np.indices((12,2)).transpose(1,2,0).reshape(24,2)
    Dfp_buf = np.apply_along_axis(lambda v: mutual_info(f_bar[:,v[0]], p_bar[:,v[1]]), 1, Dfp_buf).reshape(12,2)
    Dfp_buf = np.sum(Dfp_buf, axis=1)

    Wfr_buf = Dfr_buf / (Dfr_buf + Dfp_buf + epsilon)
    tot = np.sum(Dcr_buf) + np.sum(Dcp_buf) + np.sum(Dcf_buf)
    Wcf_buf = Dfc_buf / tot
    
    beta_buf = Wcf_buf * (1 - Wfr_buf)

    return alpha_buf, beta_buf, Wcr_buf



# %%
def calc_color(index):
    pass
    

def filterColor(x, y, N, P, indexN, setColor, alpha, beta, Wcr):
    v = 0.002
    vc = v / (1 - np.mean(Wcr)) ** 2
    vf = vc
    coeff_c = -0.5 * (1 / vc)
    coeff_f = -0.5 * (1 / vf)

    # FEATURECOUNT = 12
    
    # c_s = np.zeros((spp, 3))

    # for index, i in enumerate(P):
    #     c_pp = np.zeros(3)
    #     w = 0
        
    #     ci_bar = N[index, 2:5]
    #     fi_bar = N[index, 2:14]
    #     for indexj, j in enumerate(N):
    #         s1 = 0.0
    #         s2 = 0.0
            
    #         cj_bar = j[2:5]
    #         fj_bar = j[2:14]

    #         s1 = np.sum(alpha * ((ci_bar - cj_bar) ** 2))
    #         s2 = np.sum(beta * ((fi_bar - fj_bar) ** 2))
    #         wij = math.exp(coeff_c * s1) * math.exp(coeff_f * s2)
            

    #         c_pp += wij * color[indexN[indexj][0]][indexN[indexj][1]][indexN[indexj][2]]
    #         w += wij
            
    #     c_pp = (1 / w) * c_pp
    #     c_s[index] = c_pp
    # buffer[x, y] = c_s
    
    
    sample_cnt = N.shape[0]
    
    ci_mat = np.repeat(N[0:spp, 2:5], repeats=sample_cnt, axis=0)
    cj_mat = np.tile(N[:, 2:5], (spp,1))
    # alpha_mat = np.repeat(alpha, repeats=sample_cnt * spp, axis=0).reshape(sample_cnt * spp, 3)
    alpha_mat = np.tile(alpha, (spp * sample_cnt, 1))
    
    s1ij = np.sum(alpha_mat * ((ci_mat - cj_mat) ** 2), axis=1).reshape(spp, sample_cnt)
    
    fi_mat = np.repeat(N[0:spp, 2:14], repeats=sample_cnt, axis=0)
    fj_mat = np.tile(N[:, 2:14], (spp,1))
    beta_mat = np.tile(beta, (spp * sample_cnt, 1))
    s2ij = np.sum(beta_mat * ((fi_mat - fj_mat) ** 2), axis=1).reshape(spp, sample_cnt)
    wij = np.exp(coeff_c * s1ij + coeff_f * s2ij)
    
    row_sums = wij.sum(axis=1)
    wij = wij / row_sums[:, np.newaxis]
    
    buffer[x, y] = wij @ setColor
    
    
    
    
    

# %%
def render(file, c):
    colors = np.zeros([width, height, 3])
    colors[:,:,2] = c[:,:,0:8,0].mean(-1)
    colors[:,:,1] = c[:,:,0:8,1].mean(-1)
    colors[:,:,0] = c[:,:,0:8,2].mean(-1)
    
    cv.imwrite(file, colors * 255.0)

# %%
# boxSizes = [55, 35, 17, 7]
# boxSizes = [5, 3]
# boxSizes = [10, 3]
boxSizes = [10, 7, 5, 3]

color = samples[:,:,:,2:5]
buffer = np.copy(color)

print(np.max(color - samples[:,:,:,2:5]))

for t in range(len(boxSizes)):
    b = boxSizes[t]
    maxNumOfSamples = (b * b * spp) // 2
    for i in range(height):
        pre = 0
        fw = 0
        fc = 0
        for j in range(width):
            # if not(j <= 0.30 * width and i <= 0.60 * height and j > 0.10 * width and i > 0.38 * height):
                # continue
            tmp_c = time.time()
            N, P, indexN, setColor = preprocess(i, j, b, maxNumOfSamples)
            tmp_d = time.time()
            pre += tmp_d - tmp_c
            
            alpha, beta, wcr = computeFeatureWeights(b, N)
            tmp_c = time.time()
            fw += tmp_c - tmp_d 
            filterColor(i, j, N, P, indexN, setColor, alpha, beta, wcr)
            tmp_d = time.time()
            fc += tmp_d - tmp_c
            # break
        # break
        print("preprocess: " + str(pre) + " featureWeight: " + str(fw) + "filterColor: " + str(fc))
        print(t, i, j)
    # break
    print(np.max(color - buffer))
    print(np.argmax(color - buffer))
    render("../testres/diff-boxsize-" + str(t) +".jpg", normalize(color-buffer))
    render("../testres/boxsize-" + str(t) +".jpg", buffer)
    
    color = np.copy(buffer)
    print("=============")
    
render('../testres/color.jpg', color)
print(np.max(color - samples[:,:,:,2:5]))
print(np.argmax(color - samples[:,:,:,2:5]))
# %%
