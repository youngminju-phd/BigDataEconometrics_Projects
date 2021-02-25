import numpy as np
from timeit import default_timer as timer
from sklearn.linear_model import LinearRegression

# A.1. Inner product of two vectors:

def inner_prod(v1, v2):
    'inner production of two vectors.'
    result = 0
    for i in range(len(v1)):
        result += v1[i] * v2[i]
    return result

# defining vectors in numpy:
x = np.matrix('1;0;0;2')
y = np.matrix('1;1;3;2')
print("A.1. Inner product of two vectors")
print("Two vectors.")
print("x:")
print(x)
print("y:")
print(y)
print("Inner product of x and y:")
print(inner_prod(x,y))


# A.2. Product of two matrices:

def multiply(m, v):
    'matrix multiply vector by inner production.'
    return [inner_prod(r, v) for r in m]

# defining matrices in numpy:
#a = np.matrix([[1,0],[0,2]]) #alternatively:
a = np.matrix('1,0;0,5;4,1')
b = np.matrix([[1,0],[0,6]])

a = np.squeeze(np.asarray(a))
b = np.squeeze(np.asarray(b))

print("A.2. product of two matrices")
print("Two matrices")
print("a:")
print(a)
print("b:")
print(b)
print("Product of a and b:")
print(multiply(a,b))



# B: 1,2 m=50, n=100, p=200
print("B.")
m=50
n=100
p=200

X=np.random.random((m,n))
Y=np.random.random((n,p))

start1 = timer()

multiply(X,Y)

end1 = timer()
print('Elapsed time of the custom function for B1:',end1 - start1)

start2 = timer()

np.matmul(X,Y)

end2 = timer()
print('Elapsed time of built-in NumPy method for B2:',end2 - start2)


# C: 1,2 m=500, n=1000, p=2000
print("C.")
m=500
n=1000
p=2000

X=np.random.random((m,n))
Y=np.random.random((n,p))

start1 = timer()

multiply(X,Y)

end1 = timer()
print('Elapsed time of the custom function for C1:',end1 - start1)

start2 = timer()

np.matmul(X,Y)

end2 = timer()
print('Elapsed time of built-in NumPy method for C2:',end2 - start2)


# D: 1,2 m=500, n=1000, p=2000
print("D.")
m=5000
n=10000
p=20000

X=np.random.random((m,n))
Y=np.random.random((n,p))

start = timer()

np.matmul(X,Y)

end = timer()
print('Elapsed time of built-in NumPy method for D:',end - start)


# E: Model Intialization
print("E.")
reg = LinearRegression()

# 1. n=5000, p=200
n1=5000
p1=200
X1=np.random.rand(n1,p1)
Y1=np.random.random(n1)

# Regression
start1 = timer()
reg1 = reg.fit(X1, Y1)
end1 = timer()
print('Elapsed time of n=5000, p=200:',end1 - start1)

# 2. n=50000, p=200
n2=50000
p2=200
X2=np.random.rand(n2,p2)
Y2=np.random.random(n2)

# Regression
start2 = timer()
reg2 = reg.fit(X2, Y2)
end2 = timer()
print('Elapsed time of n=50000, p=200:',end2 - start2)

# 3. n=5000, p=2000
n3=5000
p3=2000
X3=np.random.rand(n3,p3)
Y3=np.random.random(n3)

# Regression
start3 = timer()
reg3 = reg.fit(X3, Y3)
end3 = timer()
print('Elapsed time of n=5000, p=2000:',end3 - start3)

'''
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print("RMSE")
print(rmse)
print("R2 Score")
print(r2)
print("Coefficient")
print(reg.coef_)
'''