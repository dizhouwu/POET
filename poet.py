import numpy as np
import numba
import scipy
from copy import deepcopy

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = self.__from_nested_dict(self[key])

    def __setattr__(self, key, value):
        if hasattr(dict, key) and callable(getattr(dict, key)):  # Raise error if attempting to override dict method
            raise AttributeError(f'Attempting to override dict method: {key}')
        super(DotDict, self).__setattr__(key, value)

    @classmethod
    def __from_nested_dict(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.__from_nested_dict(data[key]) for key in data})


def estimate_nfactor_act(X, C=1):
    """
    estimate number of factors given data matrix X (n*p)
    threshold on eigenvalues of correlation matrix (bias corrected)
    https://arxiv.org/abs/1909.10710
    K = # eigenvalues of sample corr that > 1 + sqrt(p / (n-1))
    """
    n, p = X.shape

    # 1. get sample correlation matrix and eigenvalues
    corr = np.corrcoef(X.T)
    evals = np.flip(np.linalg.eigvalsh(corr))  # order large to small

    # 2. get bias corrected eigenvalues
    evals_adj = np.zeros(p - 1)
    for i in range(p - 1):
        mi = (
            np.sum(1.0 / (evals[(i + 1) :] - evals[i]))
            + 4.0 / (evals[i + 1] - evals[i])
        ) / (p - i)
        rho = (p - i) / (n - 1)
        evals_adj[i] = -1.0 / (-(1 - rho) / evals[i] + rho * mi)

    # 3. threshold to estimate number of factors
    thres = 1.0 + np.sqrt(p / (n - 1)) * C
    return np.where(evals_adj > thres)[0][-1] + 1  # max_j that lambda_j > thres



sign = lambda x: x and (1 if x >0 else -1)
def POET(Y, K=-np.inf, C=-np.inf, thres='soft', matrix='cor'):
    if K == -np.inf:
        try:
            K = estimate_nfactor_act(Y)
        except IndexError:
            print("ill-formed matrix Y, provide K with suggestion (K>0 and K<=8)")
            return

    # Y: p feature * n obs
    p, n = Y.shape
    Y = Y- Y.mean(axis=1)[:, np.newaxis]

    if K==-np.inf:
        K1=0.25*(POETKhat(Y).K1HL+POETKhat(Y).K2HL+POETKhat(Y).K1BN+POETKhat(Y).K2BN)
        K=np.floor(K1)+1

    if K>0:
        Dd, V = np.linalg.eig(Y.T @ Y)
        F = np.sqrt(n)*V[:,:K]  #F is n by K
        LamPCA = Y @ F / n
        uhat = Y - LamPCA @ F.T  # p by n
        Lowrank = LamPCA @ LamPCA.T
        rate = 1/np.sqrt(p)+np.sqrt((np.log(p))/n)
    else:
        uhat = Y # Sigma_y itself is sparse
        rate = np.sqrt((np.log(p))/n)
        Lowrank = np.zeros([p,p])

    SuPCA = uhat @ uhat.T / n
    SuDiag = np.diag(np.diag(SuPCA))
    if matrix == 'cor':
        R = np.linalg.inv(SuDiag**(1/2)) @ SuPCA @ np.linalg.inv(SuDiag**(1/2))
    if matrix == 'vad':
        R = SuPCA

    if C == -np.inf:
        C = POETCmin(Y,K,thres,matrix)+0.1

    uu = np.zeros([p,p,n])
    roottheta = np.zeros([p,p])
    lambda_ = np.zeros([p,p])

    for i in range(p):
        for j in range(i): # symmetric matrix
            uu[i,j,:] = uhat[i,] * uhat[j,]
            roottheta[i,j] = np.std(uu[i,j,:],ddof=1)
            print(np.std(uu[i,j,:],ddof=1))
            lambda_[i,j] = roottheta[i,j]*rate*C
            lambda_[j,i] = lambda_[i,j]

    Rthresh = np.zeros([p,p])

    if thres == 'soft':
        for i in range(p):
            for j in range(i):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                elif j == i:
                    Rthresh[i,j] = R[i,j]
                else:
                    Rthresh[i,j]=sign(R[i,j])*(abs(R[i,j])-lambda_[i,j])
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'hard':
        for i in range(p):
            for j in range(i):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                else:
                    Rthresh[i,j] = R[i,j]
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'scad':
        for i in range(p):
            for j in range(i):
                if j == i:
                    Rthresh[i,j] = R[i,j]
                elif abs(R[i,j] < lambda_[i,j]):
                    Rthresh[i,j] = 0
                elif abs(R[i,j])<2*lambda_[i,j]:
                    Rthresh[i,j]=sign(R[i,j])*(abs(R[i,j])-lambda_[i,j])
                elif abs(R[i,j])<3.7*lambda_[i,j]:
                    Rthresh[i,j]=((3.7-1)*R[i,j]-sign(R[i,j])*3.7*lambda_[i,j])/(3.7-2)
                else:
                    Rthresh[i,j] = R[i,j]
                Rthresh[j,i] = Rthresh[i,j]

    SigmaU = np.zeros([p,p])
    if matrix == 'cor':
        SigmaU = SuDiag**(1/2) @ Rthresh * SuDiag**(1/2)
    if matrix == 'vad':
        SigmaU = Rthresh

    SigmaY = SigmaU + Lowrank

    result = DotDict({'SigmaU':SigmaU,
              'SigmaY':SigmaY,
              'factors':F.T,
              'loadings':LamPCA})
    return result


def POETCmin(Y,K,thres,matrix):
    p, n = Y.shape

    def mineig(Y,K,C,thres,matrix):
        SigmaU = POET(Y,K,C,thres,matrix).SigmaU
        f = min(np.linalg.eigvals(SigmaU))
        return f

    def f(x):
        return mineig(Y,K,x,thres,matrix)

    if f(50)*f(-50)<0:
        roots = scipy.optimize.fsolve(f,[-50,50])
        result = max(0,roots)
    else:
        result = 0
    return result


def POETKhat(Y):
    p, n = Y.shape
    Y = Y- Y.mean(axis=1)[:, np.newaxis]
    #Hallin and Liska method

    c=np.arange(0.05, 5.05,0.05)
    re=20
    rmax=10
    IC=np.zeros([2,re,rmax,100])
    gT1HL, gT2HL, pi, ni=np.ones(20),np.ones(20),np.ones(20),np.ones(20)

    for i in range(re): #generate the subsets, "re" of them
        pi[i]=min(i*np.floor(p/re)+min(p,5),p)
        ni[i]=min(i*np.floor(n/re)+min(n,5),n)
        if (i==re):
            pi[i]=p
            ni[i]=n
        Yi=Y[:pi[i],:ni[i]]
        frob=np.zeros(rmax)
        penal=np.zeros(rmax)

        for k in range(min(pi[i],ni[i],rmax)):
            Dd, V = np.linalg.eig(Yi.T @ Yi)
            F = V[:,:K]
            LamPCA = Yi @ F / ni[i]
            uhat = Yi - LamPCA @ F.T # pi by ni
            frob[k]=sum(np.diag(uhat @ uhat.T))/(pi[i]*ni[i])
            gT1HL[i]=np.log((pi[i]*ni[i])/(pi[i]+ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])
            gT2HL[i]=np.log(min(pi[i],ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])

            for l in range(100): # only fills in the ICs up to k, which may be <rmax
                IC[1,i,k,l]=np.log(frob[k])+c[l]*k*gT1HL[i]
                IC[2,i,k,l]=np.log(frob[k])+c[l]*k*gT2HL[i]


    rhat=np.zeros([2,re,100])
    for i in range(re):
        for l in range(100):
            m=min(pi[i],ni[i],rmax)
            temp1=np.argmin(IC[0,i,:m,l])
            rhat[0,i,l]=temp1
            temp2=np.argmin(IC[1,i,:m,l])
            rhat[1,i,l]=temp2

    sc1, sc2 = np.zeros(100), np.zeros(100)

    for l in range(100):
        sc1[l] = np.std(rhat[0,:,l],ddof=1)
        sc2[l] = np.std(rhat[1,:,l],ddof=1)

    c1vec=np.where(sc1==0)
    ctemp1=c1vec[0]
    c1=c[ctemp1]
    K1HL=rhat[0,0,ctemp1]

    c2vec=np.where(sc2==0)
    ctemp1=c1vec[0]
    c1=c[ctemp1]
    K1HL=rhat[0,0,ctemp1]


    c=1
    rmax=10
    IC=np.zeros([2,rmax])
    frob, penal = np.zeros(rmax), np.zeros(rmax)

    for k in range(rmax):
        Dd, V = np.linalg.eig(Y.T @ Y)
        F = V[:,:K]
        LamPCA = Y @ F / n
        uhat = Yi - LamPCA @ F.T # p by n
        frob[k]=sum(np.diag(uhat @ uhat.T))/(p*n)
        gT1BN=np.log(np.log((p*n))/(p+n))*(p+n)/(p*n)
        gT2B=np.log(min(p,n))*(p+n)/(p*n)
        IC[0,k]=np.log(frob[k]) +k*gT1BN
        IC[1,k]=np.log(frob[k]) +k*gT2BN

    K1BN = np.argmin(IC[0,:])
    K2BN = np.argmin(IC[1,:])

    result = DotDict({"K1HL":K1HL,"K2HL":K2HL,"K1BN":K1BN,"K2BN":K2BN,"IC":IC})
    return result


if __name__ == "__main__":
    mat=np.array([[-0.62029989,
      0.10368819,
      -2.6429999,
      -0.6644259,
      -0.9588529,
      -0.57635678,
      -0.21164741,
      -0.9944665,
      -0.5399032,
      -0.90196802],
     [-0.36667783,
      0.03652214,
      0.0154076,
      -0.1750231,
      1.811448,
      2.42165275,
      -0.10193561,
      0.3508604,
      -0.9005045,
      -0.75846084],
     [2.03660714,
      -0.25794352,
      -1.0779967,
      0.8037299,
      0.2745545,
      -1.22127856,
      -1.35214726,
      -0.2573654,
      0.1993435,
      1.72361045],
     [0.63555031,
      0.38537656,
      1.2166375,
      -0.9678573,
      0.5881848,
      0.90408146,
      -0.47367684,
      0.4895787,
      0.2945947,
      -1.24315542],
     [1.74910748,
      -1.80102071,
      -0.9853598,
      -2.1408104,
      0.6442657,
      0.30564053,
      0.50115736,
      0.377065,
      0.8555455,
      -0.03415805],
     [0.38047868,
      -0.31374989,
      -1.0071146,
      -3.2459573,
      -2.1209848,
      2.26800656,
      -0.6119524,
      -1.337323,
      -1.8475929,
      0.27715889],
     [-0.20937215,
      -0.54497866,
      1.6817701,
      -0.6987112,
      -1.4058039,
      -0.95944526,
      -0.58590743,
      0.9026475,
      -0.1828129,
      -0.10631256],
     [0.04250279,
      -1.15922195,
      -0.6142242,
      1.2222378,
      -1.1787129,
      1.20142,
      0.03496321,
      0.5629185,
      0.2380585,
      -0.51621383],
     [0.84506649,
      0.86538678,
      -0.1047678,
      1.5045583,
      0.5522289,
      -0.04782663,
      0.55088789,
      -0.5293272,
      -0.8919249,
      0.51648943],
     [0.77542942,
      0.14709082,
      2.5020118,
      -0.4079575,
      -0.1691369,
      -0.66425291,
      -0.70136209,
      0.159693,
      1.0314144,
      1.10792768]])
    a =POET(mat,K=3,C=0.5, thres='soft', matrix='vad')

    print(a)
