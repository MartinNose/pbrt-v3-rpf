import struct
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    with open('./scenes/sample.dat', 'rb') as f:
        fileContent = f.read()
        width, height, spp, SL = struct.unpack("LLLL", fileContent[:32])
        m = struct.unpack("f" * ((len(fileContent) - 32) // 4), fileContent[32:])
    print(width, height, spp, SL)
    # print(np.array(m[0:28]))
    index = 10 * width * spp * SL + 5 * spp * SL + 4 * SL
    print(np.array(m[index:index + 28]))
    
    raw = np.array(m)
    m = np.reshape(raw, (width, height, spp, SL))
    # print(m[0,0,0,:])
    print(m[10,5,4,:])    
    
    colors = np.zeros([width, height, 3])
    colors[:,:,2] = m[:,:,0:8,2].mean(-1)
    colors[:,:,1] = m[:,:,0:8,3].mean(-1)
    colors[:,:,0] = m[:,:,0:8,4].mean(-1)

    cv.imwrite('color.jpg', colors*255)
    cv.imshow("image", colors)
    cv.waitKey()

    exit()