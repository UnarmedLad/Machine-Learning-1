#--- 1.1 ----
import numpy as np
import matplotlib.pyplot as plt
def gen_sinusodial(N):
    width = 2*np.pi/(N-1)
    xvec = np.arange(0,2*np.pi+width,width)
    tvec = np.zeros(N)
    for i in range(len(tvec)):
        tvec[i] = np.random.normal(np.sin(xvec[i]),0.2)
    return xvec,tvec

#--- 1.2 ---

def fit_polynomial(x,t,M):
    A_feature =  make_feature_matrix(x,M)
    penn_inv = np.linalg.pinv(A_feature)
    return np.dot(penn_inv,t)
    

def make_feature_matrix(x,M):
    A_feature = np.zeros([len(x),M+1])
    (dim_1,dim_2) = A_feature.shape
    for i in range(dim_1):
        for j in range(dim_2):
            A_feature[i,j] = x[i]**j
    return A_feature
#--- 1.3 ---

N=9
x,t = gen_sinusodial(N)
M=[0,1,3,9]
for i in M:
    w_ = fit_polynomial(x,t,i)
    xcor = np.arange(0,2*np.pi,0.01)
    func = np.poly1d(w_[::-1])
    funcor = func(xcor)
    sincor = np.sin(xcor)
    plt.figure()
    plt.plot(xcor,funcor)
    plt.plot(xcor,sincor)
    plt.scatter(x,t)

#--- 1.4 ---
def fit_polynomial_reg(x,t,M,lamb):
    A_feat =  make_feature_matrix(x,M)
    waarde = np.dot(np.linalg.inv(lamb*np.eye(M+1) + np.dot(np.transpose(A_feat),A_feat)),np.transpose(A_feat))
    return np.dot(waarde,t)
#--- 1.5 ---
lamb = np.exp(-np.arange(10,0,-1))
M = range(0,11)
N=9

x,t= gen_sinusodial(N)
def model_sel_crossval(lamb,M,x,t):
    rms_matrix = np.zeros([len(lamb), len(M)])
    for i in range(len(lamb)):
        for j in range(len(M)):
            #initial stuff
            rms_mini = np.zeros([N,1])
            for k in range(N):
                trainIdx = range(N)
                del trainIdx[k]
                w = fit_polynomial_reg(x[trainIdx],t[trainIdx],M[j],lamb[i])
                func_ = np.poly1d(w[::-1])
                pred_y = func_(x[k])
                rms_mini[k] = np.sqrt(np.mean(np.square(pred_y - t[k])))
            rms_matrix[i,j] = np.mean(rms_mini)
    return rms_matrix
rms_matrix = model_sel_crossval(lamb,M,x,t)
#print rms_matrix
ind =  np.argmin(rms_matrix)
ind_ = np.unravel_index(ind,rms_matrix.shape)
print ind_[0],ind_[1]
plt.figure()
plt.matshow(np.log(rms_matrix))
#--- 1.6 ---

lamb = np.exp(-np.arange(10,0,-1))
M = range(0,11)
N=9
x,t= gen_sinusodial(N)
rms_mat =model_sel_crossval(lamb,M,x,t)
ind = np.unravel_index(np.argmin(rms_mat),rms_mat.shape)
lamb_best = lamb[ind[0]]
M_best = M[ind[1]]
w_best = fit_polynomial_reg(x,t,M_best,lamb_best)
func_best = np.poly1d(w_best[::-1])
xcor = np.arange(0,2*np.pi+0.05,0.05)
sincor = np.sin(xcor)
funcor = func_best(xcor)
plt.figure()
plt.plot(xcor,sincor)
plt.plot(xcor,funcor)
plt.scatter(x,t)
#--- 2.1 ---
def gen_sinusodial2(N):
    xvec = np.random.rand(1,N)*2*np.pi
    xvec = xvec[0]
    tvec = np.zeros(N)
    for i in range(len(tvec)):
        tvec[i] = np.random.normal(np.sin(xvec[i]),0.2)
    return xvec,tvec
#--- 2.2 ---
def fit_polynomial_bayes(x,t,M,alpha,beta):
    phi = make_feature_matrix(x,M)
    covar = np.linalg.inv(alpha*np.eye(M+1) + beta*np.dot(np.transpose(phi),phi))
    mean = beta*np.dot(np.dot(covar,np.transpose(phi)),t)
    return mean,covar

#--- 2.3 ---
def predict_polynomial_bayes(x,m,S,beta):
    phi = make_feature_vector(x,M)
    mean = np.dot(np.transpose(m),phi)
    var = 1.0/beta + np.dot(np.dot(np.transpose(phi),S),phi)
    return mean, var

def make_feature_vector(x,M):
    phi = np.zeros(M+1)
    for j in range(M+1):
        phi[j] = x**j
    return phi
#
#--- 2.4a ---
#intial settings
N=7
x,t = gen_sinusodial2(N)
M=5
alpha = 0.5
beta = 25.0 #its just 25
#compute posterior
mean,covar = fit_polynomial_bayes(x,t,M,alpha,beta)
#predict values
xcor = np.arange(0,2*np.pi+0.05,0.05)
means = np.zeros(len(xcor))
varss = np.zeros(len(xcor))
for i,val in enumerate(xcor):
    means[i],varss[i] = predict_polynomial_bayes(val,mean,covar,beta)
#plot results
plt.figure()
plt.plot(xcor,means)
plt.fill_between(xcor, means-varss,means+varss,alpha=0.1)
plt.scatter(x,t)
#--- 2.4b ---
#data is same as above
xcor = np.arange(0,2*np.pi+0.05,0.05)
plt.figure()
for i in range(100):
    w = np.random.multivariate_normal(mean,covar)
    func = np.poly1d(w[::-1])
    fcor = func(xcor)
    plt.plot(xcor,fcor)
plt.show()
