"""
Applying ivis to the MNIST Dataset
==================================

Ivis can be easily applied to unstructured datasets, including images.
Here we visualise the MNSIT digits dataset using two-dimensional ivis
embeddings.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from ivis.ivis import Ivis

mnist = fetch_openml('mnist_784', version=1)
ivis = Ivis(model='maaten', verbose=1)
<<<<<<< HEAD
n = 20000
=======
n = 70000
>>>>>>> 1c2d987cc9d9be3477aec09b7aa3151a714ffdde
data = mnist.data[:n, :]
embeddings = ivis.fit_transform(data)


color = mnist.target[:n].astype(int)

plt.figure(figsize=(8, 8), dpi=150)
plt.scatter(x=embeddings[:, 0],
            y=embeddings[:, 1], c=color, cmap="Spectral", s=1)
plt.xlabel('ivis 1')
plt.ylabel('ivis 2')
plt.show()
plt.savefig('figure.png')
os.remove('annoy.index')
