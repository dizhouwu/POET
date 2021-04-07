import numpy as np
import numba
import scipy
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

def _estimate_K(Y):
    POETKhat_y = POETKhat(Y)
    K1=0.25*(POETKhat_y.K1HL+POETKhat_y.K2HL+POETKhat_y.K1BN+POETKhat_y.K2BN)
    K=np.floor(K1)+1
    return K
    
def POET(Y, K=-np.inf, C=-np.inf, thres='soft', matrix='cor'):
    """
    Estimates large covariance matrices in approximate factor models by thresholding principal orthogonal complements.

    Y:p by n matrix of raw data, where p is the dimensionality, n is the sample size. It is recommended that Y is de-meaned, i.e., each row has zero mean
    K: number of factors
    C: float, the positive constant for thresholding.C=0.5 performs quite well for soft thresholding
    thres: str, choice of thresholding. K=0 corresponds to threshoding the sample covariance directly
    matrix: the option of thresholding either correlation or covairance matrix.'cor': threshold the error correlation matrix then transform back to covariance matrix. 'vad': threshold the error covariance matrix directly.

    Return:
    SigmaY: estimated p by p covariance matrix of y_t
    SigmaU: estimated p by p covariance matrix of u_t
    factors: estimated unobservable factors in a K by T matrix form
    loadings: estimated factor loadings in a p by K matrix form
    """
    if K == -np.inf:
        try:
            K = estimate_nfactor_act(Y)
        except IndexError:
            print("ill-formed matrix Y, provide K with suggestion (K>0 and K<8)")
            return
        
#     if K==-np.inf:
#         K=_estimate_K(Y)

    # Y: p feature * n obs
    p, n = Y.shape
    Y = Y- Y.mean(axis=1)[:, np.newaxis]

    if K>0:
        Dd, V = np.linalg.eigh(Y.T @ Y)
        Dd = Dd[::-1]
        V = np.flip(V,axis=1)
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
        for j in range(i+1): # symmetric matrix
            uu[i,j,:] = uhat[i,] * uhat[j,]
            roottheta[i,j] = np.std(uu[i,j,:],ddof=1)
            lambda_[i,j] = roottheta[i,j]*rate*C
            lambda_[j,i] = lambda_[i,j]

    Rthresh = np.zeros([p,p])

    if thres == 'soft':
        for i in range(p):
            for j in range(i+1):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                elif j == i:
                    Rthresh[i,j] = R[i,j]
                else:
                    Rthresh[i,j]=sign(R[i,j])*(abs(R[i,j])-lambda_[i,j])
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'hard':
        for i in range(p):
            for j in range(i+1):
                if np.abs(R[i,j]) < lambda_[i,j] and j < i:
                    Rthresh[i,j] = 0
                else:
                    Rthresh[i,j] = R[i,j]
                Rthresh[j,i] = Rthresh[i,j]

    elif thres == 'scad':
        for i in range(p):
            for j in range(i+1):
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
    """
    This function is for determining the minimum constant in the threshold that guarantees the positive
definiteness of the POET estimator.
    """
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
    """
    This function is for calculating the optimal number of factors in an approximate factor model.
    """
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
        if i==re-1:
            pi[i]=p
            ni[i]=n

        Yi=Y[:int(pi[i]),:int(ni[i])]
        frob=np.zeros(rmax)
        penal=np.zeros(rmax)

        for k in range(min(int(pi[i]),int(ni[i]),rmax)):
            Dd, V = np.linalg.eigh(Yi.T @ Yi)
            Dd = Dd[::-1]
            V = np.flip(V,axis=1)
            F = V[:,:k+1]
            LamPCA = Yi @ F / ni[i]
            uhat = Yi - LamPCA @ (F.T) # pi by ni

            frob[k]=sum(np.diag(uhat @ (uhat.T)))/(pi[i]*ni[i])
            gT1HL[i]=np.log((pi[i]*ni[i])/(pi[i]+ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])
            gT2HL[i]=np.log(min(pi[i],ni[i]))*(pi[i]+ni[i])/(pi[i]*ni[i])

            for l in range(100): # only fills in the ICs up to k, which may be <rmax
                IC[0,i,k,l]=np.log(frob[k])+c[l]*(k+1)*gT1HL[i]
                IC[1,i,k,l]=np.log(frob[k])+c[l]*(k+1)*gT2HL[i]


    rhat=np.zeros([2,re,100])
    for i in range(re):
        for l in range(100):
            m=min(pi[i],ni[i],rmax)
            temp1=np.argmin(IC[0,i,:int(m),l])
            rhat[0,i,l]=temp1
            temp2=np.argmin(IC[1,i,:int(m),l])
            rhat[1,i,l]=temp2

    rhat+=1
    sc1, sc2 = np.zeros(100), np.zeros(100)

    for l in range(100):
        sc1[l] = np.std(rhat[0,:,l],ddof=1)
        sc2[l] = np.std(rhat[1,:,l],ddof=1)

    c1vec=np.where(sc1==0)
    ctemp1=c1vec[0][0]

    c1=c[ctemp1]
    K1HL=rhat[0,0,ctemp1]

    c2vec=np.where(sc2==0)
    ctemp2=c2vec[0][0]
    c2=c[ctemp2]
    K2HL=rhat[1,0,ctemp2]

    c=1
    rmax=10
    IC=np.zeros([2,rmax])
    frob, penal = np.zeros(rmax), np.zeros(rmax)

    for k in range(rmax):
        Dd, V = np.linalg.eigh(Y.T @ Y)
        Dd = Dd[::-1]
        V = np.flip(V,axis=1)
        F = V[:,:k+1]
        
        LamPCA = Y @ F / n
        uhat = Y - LamPCA @ (F.T) # p by n
        frob[k]=sum(np.diag(uhat @ uhat.T))/(p*n)
        gT1BN=np.log(np.log((p*n))/(p+n))*(p+n)/(p*n)
        gT2BN=np.log(min(p,n))*(p+n)/(p*n)
        IC[0,k]=np.log(frob[k]) +(k+1)*gT1BN
        IC[1,k]=np.log(frob[k]) +(k+1)*gT2BN


    K1BN = np.argmin(IC[0,:])
    K2BN = np.argmin(IC[1,:])

    result = DotDict({"K1HL":K1HL,"K2HL":K2HL,"K1BN":K1BN,"K2BN":K2BN,"IC":IC})
    return result



if __name__ == "__main__":
    mat=np.array([
[0.8841665, -0.2017119 , 0.7010793 ,-0.8378639],
[-0.2017119,  2.2415674 ,-0.9365252 , 1.8725689],
[ 0.7010793 ,-0.9365252 , 1.7681529 ,-0.6699727],
[-0.8378639  ,1.8725689, -0.6699727 , 2.5185530],
    ])
    a =POET(mat,K=3,C=0.5, thres='soft', matrix='vad')

