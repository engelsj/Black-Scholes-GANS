from PIL import Image
import numpy as np
import imageio

ipath = "plots/iterations/iteration_%d.png"

images = []
for i in range(11):
    images.append(imageio.imread(ipath%(i*1000)))
imageio.mimsave('plots/iterations/iterations.gif', images, fps=2)