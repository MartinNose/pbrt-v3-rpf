# %%
import numpy as np
import struct
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import scale
import cv2 as cv
from math import floor, ceil
import math

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
    N = scale(N)
    
    return N, P, indexN

# %%
def sumD(Dxx):
    return Dxx[0] + Dxx[1] + Dxx[2]

def mutual_info(x, y):
    c_xy = np.histogram2d(x, y)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def computeFeatureWeights(t, N):
    cn = [2, 3, 4] # R, G, B
    rn = [0, 1, 23, 24] # X, Y, U, V
    pn = [0, 1] # X, Y

    FEATURECOUNT = 12
    # f = N[:,2:14]
    f = list(range(2, 14))
    
    Dcr = [0.0, 0.0, 0.0] # Deps between color and random
    Dcp = [0.0, 0.0, 0,0] # Deps between color and position
    Dcf = [0.0, 0.0, 0.0] # Deps between color and feature
    Wcr = [0.0, 0.0, 0.0] # 
    epsilon = 1e-10
    alpha = [0.0, 0.0, 0.0]
    
    Dfc = np.zeros(FEATURECOUNT)
    for k in range(3):
        for j in range(4):
            Dcr[k] += mutual_info(N[:,cn[k]], N[:,rn[j]])
        for l in range(2):
            Dcp[k] += mutual_info(N[:,cn[k]], N[:,pn[l]])
        for l in range(12):
            tmp = mutual_info(N[:,cn[k]],N[:,f[l]])
            Dcf[k] += tmp
            Dfc[l] += tmp

        Wcr[k] = Dcr[k] / (Dcr[k] + Dcp[k] + epsilon)
        alpha[k] = max(1 - (1 + 0.1 * t) * Wcr[k], 0)
        
    Dfr = np.zeros(FEATURECOUNT)
    Dfp = np.zeros(FEATURECOUNT)
    Wfr = np.zeros(FEATURECOUNT)
    Wfc = np.zeros(FEATURECOUNT)
    beta = np.zeros(FEATURECOUNT)
    
    for k in range(FEATURECOUNT):
        for l in range(4):
            Dfr[k] += mutual_info(N[:,f[k]], N[:,rn[l]])
        for l in range(2):
            Dfp[k] += mutual_info(N[:,f[k]], N[:,pn[l]])
        
        Wfr[k] = Dfr[k] / (Dfr[k] + Dfp[k] + epsilon)

        Wfc[k] = Dfc[k] / (sumD(Dcr) + sumD(Dcp) + sumD(Dcf))
        beta[k] = max(1 - (1 + 0.1 * t) * Wfr[k], 0)
        
    return alpha, beta, Wcr



# %%
def filterColor(x, y, N, P, indexN, alpha, beta, Wcr):
    v = 0.002
    vc = v / (1 - np.mean(Wcr)) ** 2
    vf = vc
    FEATURECOUNT = 12
    for index, i in enumerate(P):
        c_pp = np.zeros(3)
        w = 0
        ci = i[2:5]
        fi = i[2:14]
        
        ci_bar = N[index, 2:5]
        fi_bar = N[index, 2:14]
        for indexj, j in enumerate(N):
            s1 = 0.0
            s2 = 0.0
            cj_bar = j[2:5]
            fj_bar = j[2:14]
            for k in range(3):
                s1 += alpha[k] * ((ci_bar[k] - cj_bar[k]) ** 2)
            for m in range(FEATURECOUNT):
                s2 += beta[m] * ((fi_bar[m] - fj_bar[m]) ** 2)

            wij = math.exp(-0.5 * (1 / vc) * s1) * math.exp(-0.5 * (1 / vf) * s2)

            c_pp = c_pp + wij * color[indexN[indexj][0]][indexN[indexj][1]][indexN[indexj][2]]
            w += wij

        c_pp = (1 / w) * c_pp
        buffer[x, y, index] = c_pp

# %%
def render(file, c):
    colors = np.zeros([width, height, 3])
    colors[:,:,2] = c[:,:,0:8,0].mean(-1)
    colors[:,:,1] = c[:,:,0:8,1].mean(-1)
    colors[:,:,0] = c[:,:,0:8,2].mean(-1)
    
    cv.imwrite(file, colors * 255.0)

# %%
# boxSizes = [55, 35, 17, 7]
boxSizes = [10, 7, 5, 3]
# boxSizes = [5, 3]
# boxSizes = [10, 3]

color = samples[:,:,:,2:5]
buffer = np.copy(color)

print(np.max(color - samples[:,:,:,2:5]))

for t in range(len(boxSizes)):
    b = boxSizes[t]
    maxNumOfSamples = (b * b * spp) // 2
    for i in range(height):
        for j in range(width):
            N, P, indexN = preprocess(i, j, b, maxNumOfSamples)
            alpha, beta, wcr = computeFeatureWeights(b, N)
            filterColor(i, j, N, P, indexN, alpha, beta, wcr)
        print(t, i, j)
    print(np.max(color - buffer))
    print(np.argmax(color - buffer))
    render("../testres/boxsize-" + str(t) +".jpg", normalize(color-buffer))
    
    color = np.copy(buffer)
    print("=============")
    
render('../testres/color.jpg', color)
print(np.max(color - samples[:,:,:,2:5]))
print(np.argmax(color - samples[:,:,:,2:5]))
# %%
