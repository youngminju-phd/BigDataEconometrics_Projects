import numpy as np
import matplotlib.pyplot as plt #imports library for plotting
from sklearn.decomposition import PCA #imports PCA function from sklearn.decomposition library
from sklearn.datasets import fetch_olivetti_faces


# 1. Load the Olivetti Faces data
data = fetch_olivetti_faces()

# Calculate the number of rows and columns of data
n=data.data.shape[0]
p=data.data.shape[1]
'''
# Print the all images
fig = plt.figure(figsize=(6, 6))
for i in range(400):
    ax = fig.add_subplot(20, 20, i + 1, xticks=[], yticks=[])
    ax.imshow(data.data[i].reshape(data.images[0].shape),cmap=plt.cm.bone)
plt.show()
'''


# 2. Demean each face
mean=data.data.mean(axis=0)
de_data=data.data-mean

# Print mean face
plt.imshow(mean.reshape(data.images[0].shape),cmap=plt.cm.bone)
plt.title('Mean face')
plt.show()


# 3. Compute eigenfaces
pca = PCA() # Creates PCA object
pca.fit(de_data) # Runs the PCA

# Print 9 eigenfaces
fig = plt.figure(figsize=(6, 7))
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    ax.set_title('%s Eigenface' %(i+1))
    ax.imshow(pca.components_[i].reshape(data.images[0].shape),cmap=plt.cm.bone)
plt.show()


# 4. Full image recovered
# Cumulative sum of variance explained with [n] features
var=np.cumsum(pca.explained_variance_ratio_*100)
print('Cumulative sum of variance explained \n',var)

var1=[]
var1.append(0)
var1.extend(var)
# Check how many PC do we need to keep 100% of the total variance
n_100=0
for i in range(p+1):
    if var1[i]>=100:
        break
    else:
        n_100+=1

# Plot Variance Explained
plt.ylabel('% Variance Explained')
plt.xlabel('# of Eigenfaces')
plt.title('Face recognition')
plt.ylim(0,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var1)
plt.axvline(x=n_100,color='g')
plt.axhline(y=100,color='r')
plt.show()

print('We need %s eigenfaces for recovering the full images. \n' %n_100)

# Reconstruction steps
pca_vectors=pca.components_
steps=[i for i in range(1, n, 50)]
steps.append(n_100)
E = []
for i in range(len(steps)):
	numEvs = steps[i]
	P = np.dot(data.data[0,:]-mean,pca_vectors[0:numEvs,:].T) # Projection
	R = np.dot(P,pca_vectors[0:numEvs,:])+mean # Reconstruction
	E.append(R)

# Print recovered images
fig = plt.figure(figsize=(12, 5))
for i in range(9):
    ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
    ax.set_title('Use %s Eigenfaces' %(steps[i]))
    ax.imshow(E[i].reshape(data.images[0].shape),cmap=plt.cm.bone)
ax = fig.add_subplot(2, 5, 10, xticks=[], yticks=[])
ax.set_title('Original image')
ax.imshow(data.data[0].reshape(data.images[0].shape),cmap=plt.cm.bone)
plt.show()
