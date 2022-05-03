import torch
import numpy as np
import scipy.linalg
import numpy.linalg
torch.manual_seed(28)
numpy.random.seed(28)

## This function is used for unitary U, where A is equal to log(U) and is asymmetry
## Written by YiBin Guo. @IOP, CAS. 2019.08.15
## For non-diagonal matrix
def AsymLogm(U):
    dimU = U.size()[1]
    U = U.numpy()

    ## Choose new gauge to satisfy det(U) = 1
    if( np.linalg.det(U) < 0 ):
        Idp = np.eye(dimU)
        Idp[0, 0] = -1
        U = U @ Idp
#        print('det(U) = -1, need to operate gauge transformation')

    ## Choose new gauge to satisfy asymmetry condition
    h = 0
    Idp = np.eye(dimU)
#    print(np.linalg.det(U))
#    input()

    while(1):
        h = h+1
        U = U @ Idp
        [E, C] = np.linalg.eig(U)
        Idp = np.eye(dimU)
#    print(np.linalg.det(U))
#        print('E:', E)
        k = 0;
        m = int(round(10 * np.random.rand(1)[0]))
        n = int(round(m * np.random.rand(1)[0]))
#        print('m:', m)
        
        for j in range(dimU):
            if( (np.real(E[j]) < (-1 + 1E-6)) and (abs(np.imag(E[j]) < 1E-6)) ):
#            if( (np.real(E[j]) < (-1 + 1E-8)) and ((np.imag(E[j]) < 1E-8)) ):
#            if( (abs(np.real(E[j])) < (1-1E-8)) and (abs(np.imag(E[j])) < 1E-12) ):
#            if( (np.real(E[j]) < (-1 + 1E-16))):
#                Idp[(j+k-1)%dimU, (j+k-1)%dimU] = -1
                Idp[(j+k+m*k)%dimU, (j+k+m*k)%dimU] = -1
                Idp[(j+k+m*k+m*n)%dimU, (j+k+m*k+m*n)%dimU] = -1
                k = k+1
#        print(k)
#        print(np.linalg.det(Idp))
#        print(np.linalg.det(U))
#        print(np.linalg.det(U@Idp))
#        input()

        if(k == 0):
            break


    A = scipy.linalg.logm(U)
    temA = np.zeros([dimU, dimU])
    for j in range(dimU):
        for k in range(j):
            temA[k, j] = np.real(A[k, j])

#    print('U:', U)
#    print('A:', A)
#    print('temA:', temA)
#    print(scipy.linalg.expm(temA-temA.T))
    A = temA
    
    test = scipy.linalg.expm(A-A.T) - U
#    print('Adiff:', numpy.linalg.norm(test))

#    if (np.max(test) > 1.0E-14):
 #       print('max:', np.max(test))
 #       print('norm:', numpy.linalg.norm(test))

    U = torch.from_numpy(U)
    A = torch.from_numpy(A)
    return U, A




if __name__=='__main__':
    """这里函数返回的U和输入的U第一列差一个负号"""
    U = torch.rand( 4, 4, dtype = torch.float64)
    Q, R = torch.qr(U)
    print('Q*Q.T:\n',Q@Q.t())
    U, A = AsymLogm(Q)
    U = U.numpy()
    A = A.numpy()
    print('U*U.T:\n',U@U.T)
    print('Q:\n',Q)
    print('U:\n',U)
    print('A:\n', A)
    test = scipy.linalg.expm(A-A.T) - U
    print('max:', np.max(test))
    print('norm:', numpy.linalg.norm(test))
    print('Q,U norm:', torch.norm(Q-U).item())
