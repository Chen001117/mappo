import imageio
import numpy as np

np.random.seed(2)
image = np.zeros([512,512,3]).astype(np.uint8)
image[0, :] = 1
image[:, 0] = 1
image[-1,:] = 1
image[:,-1] = 1
for i in range(20):
    x1 = np.random.randint(512)
    x2 = x1 + min(np.random.randint(50,100), 512-x1)
    y1 = np.random.randint(512)
    y2 = y1 + min(np.random.randint(50,100), 512-y1)
    image[x1:x2,y1:y2] = 1
imageio.imwrite("../envs/mujoco/assets/map.png", image)
