import numpy as np
from numpy.linalg import svd, eig
from skimage import io
import sys, os

folder, picture = sys.argv[1], sys.argv[2]

def load_imgs():

	imgs = list()
	for n in range(415):
		path = os.path.join(folder,str(n)+'.jpg')
		img = io.imread(path)
		imgs.append(img)

	imgs = np.asarray(imgs).reshape(415,600*600*3)
	
	return imgs


def eigen_face(imgs):

	imgs = imgs.astype(np.float64)
	average = np.mean(imgs, axis=0).astype(np.float64)
	
	for n in range(415):
		imgs[n] -= average

	U, S, V = np.linalg.svd(imgs.transpose(), full_matrices=False)

	return U


def reconstructure(eigen):

	path = os.path.join(folder,picture)
	img = io.imread(path).reshape(600*600*3)
	
	mean =  np.mean(imgs, axis=0)
	
	img = img - mean

	weight = np.dot(img, eigen)
		
	recon = np.zeros(600*600*3)
	for i in range(4): ##
		layer = weight[i] * eigen[:,i]
		recon += layer

	recon = recon + mean
	recon -= np.min(recon)
	recon /= np.max(recon)
	recon = (recon * 255).astype(np.uint8)
	recon = recon.reshape(600,600,3)

	io.imsave('reconstruction.jpg', recon)



if __name__ == "__main__":

	imgs = load_imgs()
	eigen = eigen_face(imgs)
	reconstructure(eigen)

	

	