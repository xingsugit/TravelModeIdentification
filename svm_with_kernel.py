import nunmpy as np
import math

## Reference: Pegasos: Primal Estimated sub-GrAdient SOlver for SVM
## Definition:
## ts: training set
## ys: ground truth of traing set
## ls: learning rate 
## linear pegasos (algorithm is in the paper fig 1.)
def linear_pegasos(ts,ys,lr,project = True):
    ## dimensions of feature space:
    d = np.shape(ts)[1]
    w = [0 for i in range(d)]
    for i in range(len(ts)):
        eta = 1/(lr*i)
	xi = ts[i]
        yi = ys[i]
        if yi*np.dot(w,xi) < 1:
            w = (1 - eta*lr)*w + eta*yi*xi
        else:
            w = (1 - eta*lr)*w
        if project:
            w = min(1, (1/np.sqrt(lr))/np.linalg.norm(w)) * w
    return w
        
##batch size: k
##maximum iterations for training
def mini_batch_pegasos(ts,ys,lr,k,T,project = True):
    d = np.shape(ts)[1]
    w = [0 for i in range(d)]
    for t in range(T):
        indexes = np.random.choice(len(ts), k)
        ## note: to select elements using a list of index, the original array type needs to be an np array
        xs = ts[indexes]
        yss = ys[indexes]
        eta = 1/(lr*t)
        for i in range(k):
            xi = xs[i]
            yi = yss[i]
            if yi * np.dot(w,xi) < 1:
                w = (1 - eta * lr) * w + eta/k * yi * xi
        if project:
            w = min(1, (1/np.sqrt(lr))/np.linalg.norm(w)) * w
    return w

def kernel_function(kernel_name,xi,xj,d,sigma):
    if kernel_name == 'poly':
        return (np.dot(xi,xj) + 1)**d
    if kernel_name = 'gaussian':
        return np.exp(-np.linalg.norm(xi-xj) **2 /(2* sigma^2))

##  kernel pegasos
def kernalized_pegasos(ts,ys,lr):
    num = len(ts)
    alpha = [0 for i in range(num)]
    for i,xi in enumerate(ts):
        yi = ys[i]
        sum_check = 0
        for j,xj in enumerate(ts):
            if i!= j:
                yj = y[j]
                sum_check += alpha[j] * yi * kernel_function(xi,xj)
        if yi * 1/lr * sum_check < 1:
            alpha[i] = alpha[i] + 1
    #calculate the w: w = 1/(lr*num) * sum_j(alpha[j]*ys[j]*feature_vector(x))
    w = [0 for i in range(np.shape(ts)[1]]
    for j in range(num):
        yj = ys[j]
        xj = ts[j]
        w += alpha[j] * yj * fv(xj)  
    w = 1/(lr * num) * w
    return w


## todo: add multi class classification             
