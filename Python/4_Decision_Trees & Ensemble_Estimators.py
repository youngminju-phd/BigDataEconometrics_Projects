# %reset -f
import time
start_time = time.time()

##########################################################################################
# 1 Decision Trees
##########################################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt

# A 
# Importing the dataset
from sklearn.datasets import load_boston
boston = load_boston()
X, y = boston.data, boston.target #data for X and y are stored in .data and .target
n_samples, n_features = X.shape #shape (dimensions) of X
print('1-A. A Description of Data')
print(boston['DESCR']) #print a description of a dataset

##########################################################################################
# B
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Applying K-fold Cross Validation (K=5) for depths grid
cver = KFold(n_splits = 5, shuffle = True, random_state = 1234)
folds = tuple(cver.split(X, y))

# estimation environment object:
GridSearch = lambda estimator, estimator_params: GridSearchCV(estimator, estimator_params, cv=folds, scoring='neg_mean_squared_error', return_train_score=True)

tree = DecisionTreeRegressor(random_state=1234)
tree_params = {'max_depth': range(1, 9)} #range for max depth of a tree
tree_cv = GridSearch(tree, tree_params) #creates a tree estimator object
tree_cv.fit(X, y) #fits a tree to X,y
tree_cv_results = pd.DataFrame(tree_cv.cv_results_) #estimation results transformed to a pd.dataframe

#note how you can access pd.dataframe columns by names:
plt.plot(tree_cv_results['param_max_depth'], -tree_cv_results['mean_test_score'], label='test error')
plt.plot(tree_cv_results['param_max_depth'], -tree_cv_results['mean_train_score'], label='training error')
plt.title('Train Error Vs. Test Error (Decision Tree)')
plt.legend(('Train Error', 'Test Error'))
plt.xlabel('Depth of Tree')
plt.ylabel('Prediction Error')
plt.savefig('graph_1.png')
plt.show()

#from sklearn.model_selection import cross_validate
#cv_results = cross_validate(classifier, X=X, y=y, return_train_score = True, cv = 5)
##########################################################################################

# C

# The optimal maximum depth
# optimal "score" (i.e. negative MSE) calculated over the testing sample
print('1-C. Optimal Maximum Depth')
depth_opt = tree_cv_results.param_max_depth[tree_cv_results['rank_test_score']==1]
print('Optimal Maximum Depth is', depth_opt.to_string(index=False))


##########################################################################################
# 2 Ensemble Estimators
##########################################################################################

# A

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

Bgrid = np.linspace(1, 200, 200) # the number of trees aggregated
n_B = len(Bgrid)
depthcandidate = [2,6]
test_err = np.zeros((n_B, len(depthcandidate)))

for j in range(len(depthcandidate)):
    dep = depthcandidate[j]
    for i in range(n_B):
        estimator = BaggingRegressor(base_estimator = DecisionTreeRegressor(criterion = "mse", random_state = 1234, max_depth = dep), n_estimators = i+1)
        err = cross_val_score(estimator = estimator, X = X, y = y, cv = folds, scoring = 'neg_mean_squared_error')
        test_err[i,j] = -np.mean(err)

plt.figure(figsize=(10,10))        
plt.plot(Bgrid, test_err[:,0], color = 'blue')
plt.plot(Bgrid, test_err[:,1], color = 'red')
plt.title('Test Error for Depth 2 and 6 (Bagging)')
plt.xlabel('Number of Aggregated Trees')
plt.ylabel('Prediction Error')
plt.legend(('Depth = 2', 'Depth = 6'))
plt.savefig('graph_2.png')
plt.show()      
    
##########################################################################################

# B

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

Bgrid = np.linspace(1, 200, 200) # the number of trees aggregated
n_B = len(Bgrid)
depthcandidate = [2,6]
test_err = np.zeros((n_B, len(depthcandidate)))

for j in range(len(depthcandidate)):
    dep = depthcandidate[j]
    for i in range(n_B):
        estimator = RandomForestRegressor(criterion = "mse", random_state = 1234, max_depth = dep, n_estimators = i+1)
        err = cross_val_score(estimator = estimator, X = X, y = y, cv = folds, scoring = 'neg_mean_squared_error')
        test_err[i,j] = -np.mean(err)

plt.figure(figsize=(10,10))        
plt.plot(Bgrid, test_err[:,0], color = 'blue')
plt.plot(Bgrid, test_err[:,1], color = 'red')
plt.title('Test Error for Depth 2 and 6 (Random Forest)')
plt.xlabel('Number of Aggregated Trees')
plt.ylabel('Prediction Error')
plt.legend(('Depth = 2', 'Depth = 6'))
plt.savefig('graph_3.png')
plt.show()      

##########################################################################################

# C

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

Bgrid = np.linspace(1, 200, 200) # the number of trees aggregated
n_B = len(Bgrid)
depthcandidate = [2,6]
test_err = np.zeros((n_B, len(depthcandidate)))

for j in range(len(depthcandidate)):
    dep = depthcandidate[j]
    for i in range(n_B):
        estimator = XGBRegressor(random_state = 1234, max_depth = dep, n_estimators = i+1)
        err = cross_val_score(estimator = estimator, X = X, y = y, cv = folds, scoring = 'neg_mean_squared_error')
        test_err[i,j] = -np.mean(err)

plt.figure(figsize=(10,10))        
plt.plot(Bgrid, test_err[:,0], color = 'blue')
plt.plot(Bgrid, test_err[:,1], color = 'red')
plt.ylim([10, 35])
plt.title('Test Error for Depth 2 and 6 (XGBoost)')
plt.xlabel('Number of Aggregated Trees')
plt.ylabel('Prediction Error')
plt.legend(('Depth = 2', 'Depth = 6'))
plt.savefig('graph_4.png')
plt.show()

plt.figure(figsize=(10,10))        
plt.plot(Bgrid, test_err[:,0], color = 'blue')
plt.plot(Bgrid, test_err[:,1], color = 'red')
plt.title('Test Error for Depth 2 and 6 (XGBoost)')
plt.xlabel('Number of Aggregated Trees')
plt.ylabel('Prediction Error')
plt.legend(('Depth = 2', 'Depth = 6'))
plt.savefig('graph_4_1.png')
plt.show()

##########################################################################################

elapsed_time = time.time() - start_time
print('Elapsed time is', elapsed_time)

