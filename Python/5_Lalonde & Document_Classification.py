# Econ599 HW3
# Youngmin Ju

##########################################################################################
# 1. Lalonde NSW Data
##########################################################################################

# A. Load Data
print('##############################')
print('A. Load Data')
# Import the library
from causalinference import CausalModel
from causalinference import utils

# Load the Lalonde experimental dataset
lalonde = utils.lalonde_data()

# Y = outcome, D = treatment, X = covariates
Y = lalonde[0]
D = lalonde[1]
X = lalonde[2]

# Summary statistics
causal = CausalModel(Y, D, X)

#causal.reset()
print(causal.summary_stats)
print('The largest normalized difference is ' ,max(abs(causal.summary_stats['ndiff'])))
print('X4 = Nodegree')
# max(abs(Nor-diff)) = X4 having 0.304 = abs(-0.304)

# B. Propensity score
print('##############################')
print('B. Propensity score')
causal.est_propensity_s(lin_B = [6,7,8,9])
print(causal.propensity)
# Additionally, X1, X4, X5, X4*X5, X6*X4, X9*X5

# C. Trimming
print('##############################')
print('C. Trimming')
causal.trim_s()
print('Selected cut-off is ', format(causal.cutoff, '.4f'))
# Selected cut-off is  0.1310
print(causal.summary_stats)
# Dropped 5 and 3 observations from the control and treated group respectively.
print('Dropped 5 and 3 observations from the control and treated group respectively.')

# D. Stratify
print('##############################')
print('D. Stratify')
causal.stratify_s()
print(causal.strata)
# Three propensity bins are created.
print('Three propensity bins are created.')
for stratum in causal.strata:
    stratum.est_via_ols(adj = 1)
print([stratum.estimates['ols']['ate'] for stratum in causal.strata])

# E. Estimate ATE
print('##############################')
print('E. Estimate ATE')
causal.est_via_ols()
#causal.est_via_weighting(rcond=None)
causal.est_via_blocking()
causal.est_via_matching(matches = 2, bias_adj = True)
print(causal.estimates)


##########################################################################################
# 2. Document Classification
##########################################################################################

from sklearn.datasets import fetch_20newsgroups

# A. Print out a couple sample posts
print('######################################')
print('A. Print out a couple sample posts')
newsgroups_train = fetch_20newsgroups(subset='train')

from pprint import pprint
print('######################################')
print('Print out a couple sample posts')
print('######################################')
pprint(list(newsgroups_train.data[0:2])) # a couple sample post
print('######################################')
print('List out all the topic names')
print('######################################')
pprint(list(newsgroups_train.target_names)) # topic names

# B. Convert the posts
print('######################################')
print('B. Convert the posts')
from sklearn.feature_extraction.text import TfidfVectorizer
# collect all categories
categories = [i for i in list(newsgroups_train.target_names)]
# only select the training_dataset
newsgroups_train = fetch_20newsgroups(subset = 'train', categories = categories)
# create a vectorizer to convert texts into vectors
vectorizer = TfidfVectorizer()
# fitting our vectorizer to training_dataset
vectors = vectorizer.fit_transform(newsgroups_train.data)
print('Dimensionality of vectors is: ', vectors.shape) # (11314, 130107)
print('Non-zero components are: ', vectors.nnz) # 1787565
print('The number of words is: ', vectors.nnz / float(vectors.shape[0])) # 1787565/11314=157.999

# C. #kernel PCA
print('######################################')
print('C. #kernel PCA')
# TruncatedSVD (LSA = Latent Semantic Analysis)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
# define truncatedSVD classifier (K = 30 as suggested)
svd = TruncatedSVD(n_components = 30)
# we need a normalizer since truncatedSVD requires a normalization similar to PCA
normalizer = Normalizer(copy = False)
# make lsa as new classifier conduct both svd and normalizer as defined above
lsa = make_pipeline(svd, normalizer)
# fitting lsa into our vectors
vectors = lsa.fit_transform(vectors)

# D. Supervised learning
print('######################################')
print('D. Supervised learning')
# Create and fit a nearest-neighbor classifier
print('Using KNN (k nearest neighbors) classification')
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(vectors, newsgroups_train.target)
# Predictions using vectors
print('Predictions')
# vectors: the vectorized representation of the post I obtained C.
print(knn.predict(vectors))
# Print Target
print('Target')
print(newsgroups_train.target)

# E. Tune the model
print('######################################')
print('E. Tune the model')
from sklearn import metrics
import numpy as np
# only select a testing dataset
newsgroups_test = fetch_20newsgroups(subset = 'test', categories = categories)
# vectorize testing dataset
vectors_test = vectorizer.transform(newsgroups_test.data)
# make a prediction through nearest-neighbor
knn.fit(vectors_test, newsgroups_test.target)
predictions_test = knn.predict(vectors_test)
# identify accuracy without tuning K
metrics.accuracy_score(newsgroups_test.target, predictions_test, normalize = True)
print('Accuracy w/o tuning K and w/ dimensionality reduction: ', 
      metrics.accuracy_score(newsgroups_test.target, predictions_test, normalize = True))

# Tuning the model by using test data (picking optimal K)
# re-draw test_set
newsgroups_test = fetch_20newsgroups(subset = 'test', categories = categories)
# re-vectorize test_set
vectors_test = vectorizer.transform(newsgroups_test.data)

# code that identifies optimal K (while loop)
k0 = 1 # initialization (This is desired answer)
klimit = 100 # limit of K
score=[]
while k0<=klimit:
    n_components = k0
    vectors_test = vectorizer.transform(newsgroups_test.data)
    svd = TruncatedSVD(n_components = int(n_components))
    normalizer = Normalizer(copy = False)
    lsa = make_pipeline(svd, normalizer)
    vectors_test = lsa.fit_transform(vectors_test)
    knn.fit(vectors_test, newsgroups_test.target)
    predictions_test = knn.predict(vectors_test)
    score1=metrics.accuracy_score(newsgroups_test.target, predictions_test, normalize=True)
    score.append(score1)
    k0 = k0 + 1
    print('current K: ', n_components)
    print('current score: ', score1)

value=max(score)
index=np.argmax(score)+1 # initial index = 1
print('Maximum Accuracy with optimal K and dimensionality reduction: ', value)
print('Optimal K: ', index)