import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# --------------------------------------------------
# Step 1: Ridge Regression Function 
# --------------------------------------------------

# X is the inputs (feature matrix), y is the true answers

def ridge_regression(X, y, lam):
    
# 1A: Compute X^T X

    XT = X.T
    XTX = XT @ X 
  
# 1B Compute λI (identity matrix scaled by lambda)

    # Create identity matrix for X (size 64x64)

    I = np.identity(X.shape[1])

    # Compute lambda times I (since lambda is a scalar)
    lambdaI = lam * I

# 1C Add the two matrices & invert

    XTX_plus_lambdaI = XTX + lambdaI

    # Invert the added matrices

    XTX_plus_lambdaI_inverted = np.linalg.inv(XTX_plus_lambdaI)

    
# 1D Compute X^T y

    XTY = XT @ y

# 1E Compute the final weights using values from 1C and 1D
    
    w =  XTX_plus_lambdaI_inverted @ XTY

    return(w)


# --------------------------------------------------
# Step 2: Train on dataset
# --------------------------------------------------

data = loadmat("diabetes.mat")

# 2A Extract the training and testing data
X_train = data['x_train']
y_train = data['y_train']
X_test  = data['x_test']
y_test  = data['y_test']

# 2B: Define lambda values to test
lambdas = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]

# --------------------------------------------------
# Step 3: Compute error
# --------------------------------------------------

# Prepare lists to store results
training_errors = []
testing_errors = []
weights = []
L2s = []

for lam in lambdas:
    #3A Compute optimal weights
    w = ridge_regression(X_train, y_train, lam)
    weights.append(f'Weights for lambda = {lam}:')
    for item in w:
        weights.append(float(item))

    #3B Predict y for training and testing data (y hat)
    y_train_pred = X_train @ w
    y_test_pred  = X_test @ w

    #3C Compute training and testing error
    train_error = np.mean(np.square(y_train - y_train_pred))
    test_error = np.mean(np.square(y_test - y_test_pred))

    #3D Compute L2 norm
    L2 = np.linalg.norm(w)
        

    #3E Add errors and weights to list (converted from numpy float to python float)
    training_errors.append(float(train_error))
    testing_errors.append(float(test_error))
    L2s.append(float(L2))
    
    

# --------------------------------------------------
# Step 4: Plot
# --------------------------------------------------

# 4A plot training error 
plt.plot(lambdas, training_errors)    

plt.xlabel("Lambda")
plt.xticks(lambdas)
plt.xscale('log')
plt.ylabel("Training error")
plt.title("Training error")

plt.show()

# 4B plot testing error 
plt.plot(lambdas, testing_errors)

plt.xlabel("Lambda")
plt.xticks(lambdas)
plt.xscale('log')
plt.ylabel("Testing error")
plt.title("Testing error")

plt.show()

# 4C plot L2s
plt.plot(lambdas, L2s)

plt.xlabel("Lambda")
plt.xticks(lambdas)
plt.xscale('log')
plt.ylabel("L-2 norm of weight vector")
plt.title("Weights")

plt.show()

print('Training error:', training_errors)
print('Testing error:', testing_errors)
print('L2 norms:', L2s)
print('Weights', weights)
