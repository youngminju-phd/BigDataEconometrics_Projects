# Import necessary packages

import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import zero_one_loss


# Question Two ---------------------------------------------------------

# Part One
from sklearn.datasets import load_breast_cancer
data_set = load_breast_cancer()
data = data_set['data']
target = data_set['target']

# Part Two
data_scaled = preprocessing.scale(data)  # standardizing features

# Part Three
ind = True
n_p = 1
while ind:  # a while loop for determining number of PCA for explaining 90% of total variation
    pca = PCA(n_components=n_p)  # creates pca object
    pca.fit(data_scaled)  # runs the PCA
    if np.sum(pca.explained_variance_ratio_) > 0.9:
        ind = False
    else:
        n_p += 1

print('We need ' + str(n_p) + ' principle components to explain 90% of total variation')

pca = PCA(n_components=2)  # creates pca object
pca.fit(data_scaled)  # runs the PCA
pca_vectors = pca.fit_transform(data_scaled)  # gets two principal vectors
tvar = np.sum(pca.explained_variance_ratio_)  # total variation explained by two factors

print( str(tvar*100) + ' percent of variance is explained if we keep only 2 principle components')

# Part Four

# Run k-means on all features:
km = KMeans(n_clusters=2)
km.fit(data)
km_groups = km.labels_  # predicted cluster labels
# if misclassification rate is >50%, then switch cluster labelling:
if zero_one_loss(target, km_groups) > 0.5:
    km_groups = 1-km_groups

km_loss_all = zero_one_loss(target, km_groups) # calculates misclassification rate using all features

print('Misclassification rate using all features is ' + str(km_loss_all))

# Run k-means on all scaled features:
km = KMeans(n_clusters=2)
km.fit(data_scaled)
km_groups = km.labels_  # predicted cluster labels
# if misclassification rate is >50%, then switch cluster labelling:
if zero_one_loss(target, km_groups) > 0.5:
    km_groups = 1-km_groups

km_loss_all_scaled = zero_one_loss(target, km_groups) # calculates misclassification rate using all features

print('Misclassification rate using all scaled features is ' + str(km_loss_all_scaled))

# Run k-means on first two PC:
km = KMeans(n_clusters=2)
km.fit(pca_vectors)
km_groups = km.labels_  # predicted cluster labels
# if misclassification rate is >50%, then switch cluster labelling:
if zero_one_loss(target, km_groups) > 0.5:
    km_groups = 1-km_groups

km_loss_pca = zero_one_loss(target, km_groups)  # calculates misclassification rate using pc vectors

print('Misclassification rate using first two principle components is ' + str(km_loss_pca))
