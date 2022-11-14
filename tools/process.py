import struct
import numpy as np
import cv2 as cv

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

if __name__ == '__main__':
    with open('./scenes/sample.dat', 'rb') as f:
        fileContent = f.read()
        width, height, spp, SL = struct.unpack("LLLL", fileContent[:32])
        m = struct.unpack("f" * ((len(fileContent) - 32) // 4), fileContent[32:])
    print(width, height, spp, SL)
    # print(np.array(m[0:28]))
    index = 10 * width * spp * SL + 5 * spp * SL + 4 * SL
    raw = np.array(m)
    
    m = np.reshape(raw, (width, height, spp, SL))
    
    # render color
    colors = np.zeros([width, height, 3])
    colors[:,:,2] = m[:,:,0:8,2].mean(-1)
    colors[:,:,1] = m[:,:,0:8,3].mean(-1)
    colors[:,:,0] = m[:,:,0:8,4].mean(-1)

    cv.imwrite('testres/color.jpg', colors * 255.0)

    # render position
    
    positions = np.zeros([width, height, 3])
    positions[:,:,2] = normalize(m[:,:,0:8,5].mean(-1))
    positions[:,:,1] = normalize(m[:,:,0:8,6].mean(-1))
    positions[:,:,0] = normalize(m[:,:,0:8,7].mean(-1))

    cv.imwrite('testres/position.jpg', positions * 255.0)
    
    # render normal
    
    normals = np.zeros([width, height, 3])
    normals[:,:,2] = normalize(m[:,:,0:8,8].mean(-1))
    normals[:,:,1] = normalize(m[:,:,0:8,9].mean(-1))
    normals[:,:,0] = normalize(m[:,:,0:8,10].mean(-1))

    cv.imwrite('testres/normal.jpg', normals * 255.0)

    # render texture
    
    textures = np.zeros([width, height, 3])
    textures[:,:,2] = m[:,:,0:8,11].mean(-1)
    textures[:,:,1] = m[:,:,0:8,12].mean(-1)
    textures[:,:,0] = m[:,:,0:8,13].mean(-1)

    cv.imwrite('testres/texture.jpg', textures * 255.0)
    