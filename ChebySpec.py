import numpy as np
import numpy.polynomial.chebyshev as cheb

def GLpoints(N):
    """Returns the N+1 GL points over [-1,1]
    
    Input: integer the number of Chebyshev polynomials in expansion
    
    Output: N+1 array X_j Gauss-Lobatto Chebyshev points"""
    
    X=np.zeros(N+1)
    for j in range(N+1):
        X[j]=-np.cos(np.pi*j/N)
        
    return X



def findCoeffs(U, X):
    """Find the coefficients for the expansion in a basis 
    of Chebyshev polynomials up to T_N calculated  
    at the N+1 Gauss-Lobatto points
    
    Input: N+1 array U_j=u(X_j)
           N+1 array X_j Gauss-Lobatto Chebyshev points 
    
    Output: N+1- length numpy array containing coefficients uhat
    """
    N=len(X)-1
    
    c=np.ones(N+1)
    c[0]=2
    c[-1]=2
        
    uhat=np.zeros(N+1)
    
    for k in range(N+1):
        m=2./(N*c[k])
        d=np.zeros(k+1)
        d[-1]=1
        for j in range(N+1):
            uhat[k]=uhat[k]+U[j]*cheb.chebval(X[j],d)/c[j]
        uhat[k]=m*uhat[k]
    
    return uhat