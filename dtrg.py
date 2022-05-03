import torch
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
from scipy.linalg import logm 
from itertools import product
from TorNcon import ncon
from expm import expm
import matplotlib.pyplot as plt
from AsymLogm import AsymLogm
from rig import rigfe, rigtemp

'''
本章代码为简单可微分TRG
'''


class dHOTRG(torch.nn.Module):
    

    def __init__(self, params):
        # define some basic params
        self.J = params['J']
        self.RGstep = params['RGstep']
        self.D = params['D']
        self.Torder = params['Torder']
        self.kB = params['kB']
        self.dtype = params['dtype']
        self.Temp = params['Temp']
        self.Nepochs = params['Nepochs']


    def initA(self):
        # initialize the param A-SS(skew symtric) matrix 
        # by U = exp(A)
        self.initU = []
        self.dim = []
        self.A = []
        self.Atypes = []
        self.Ashapes = []
        self.convergT = [] 
        

        T0 = self.T0
        lambdasum = 0
        for i in range(self.RGstep):
            # get the projector U
            U, A_, dim, _, _ = self.calc_gaugeU(T0)
            # if i >= self.RGstep - 2:
            #     # 只要convergT有两个T,其必然对应iniU中的最后两个U(产生自convergT中的两个T)
            #     self.convergT.append(T0)
            # contraction horizontally 
            T1, lambda_i = self.contraction(T0, U)
            # rotate the network inorder to contract it vertically
            T1 = self.rotate(T1)
            f, lambdasum = self.getPhy(T1, lambda_i, lambdasum, i+1)
            T0 = T1

            self.A.append(A_)
            self.Ashapes.append(A_.shape) # troch.size
            self.Atypes.append(A_.dtype)
            self.initU.append(U)
            self.dim.append(dim) # dtype = torch.Size
        self.lastfe = f

        return None
    
    

    def initT(self):
    # initialize the 2D Ising model tensor network
        # 2D-Ising model initial bond dimension = 2
        m = 2
        dtype = self.dtype
        beta = torch.tensor([-self.J / self.Temp], dtype=dtype)
        # Wvec is the represent for W matrix in T = WW+ 
        # (for clac simplification, Wvec is simply temporary vector to calc T)
        Wvec = torch.tensor([torch.exp(beta) + torch.exp(-beta), torch.exp(beta) - torch.exp(-beta)], dtype=dtype)
        T0 = torch.zeros(m, m, m, m, dtype=dtype)

        for i, j, k, l in product(range(m), range(m), range(m), range(m)):
            if (i + j + k + l) % 2 == 0:
                # other wise, the Tijkl = 0 , there is no need to assign values
                T0[i][j][k][l] = 0.5*torch.sqrt(Wvec[i] * Wvec[j] * Wvec[k] * Wvec[l])

        self.T0 = T0

        return None


    def calc_gaugeU(self,T):
        # calc the projector U for rewire the network
        # T:     2               U:
        #        |                    1 ---|
        #    1 ----- 3                     |---3
        #        |                    2 ---|
        #        4
        # return U
        T0 = T
        # calc the left Ul, horizontal contraction
        MMd = ncon([T0,T0,T0,T0],[[-1,2,3,4],[-5,4,6,7],[-8,2,3,9],[-10,9,6,7]],[2,3,6,7,4,9])
        dim = MMd.size()
        Matrix = MMd.flatten().reshape(dim[0] * dim[1], dim[2] * dim[3])
        El, Ul = torch.symeig(Matrix + Matrix.t(), eigenvectors=True)
        El_sorted, indices = torch.sort(El, descending=True)
        Ulshape = Ul.shape
        D = min(self.D, Ulshape[1])
        El_normed = El_sorted/El_sorted[0] 
        err_l = 1 - torch.norm(El_normed[:D])/torch.norm(El_normed)


        # calc the right Ur
        MdM = ncon([T0,T0,T0,T0],[[1,2,-8,9],[5,9,-10,7],[1,2,-3,4],[5,4,-6,7]],[1,2,5,7,4,9])
        dim = MdM.size()
        Matrix = MdM.flatten().reshape(dim[0] * dim[1], dim[2] * dim[3])
        Er, Ur = torch.symeig(Matrix + Matrix.t(), eigenvectors=True)
        Er_sorted, indices = torch.sort(Er, descending=True)
        Urshape = Ur.shape
        D = min(self.D, Urshape[1])
        Er_normed = Er_sorted/Er_sorted[0] 
        err_r = 1 - torch.norm(Er_normed[:D])/torch.norm(Er_normed)

        # compare the error and choose the one that has smaller err
        if err_r < err_l:
            U, A = AsymLogm(Ul[:,indices])
            U = U[:,:D]
            Eig = Er_sorted.clone()
            error = err_r
        else:
            U, A = AsymLogm(Ur[:,indices])
            U = U[:,:D]
            Eig = El_sorted.clone()
            error = err_l

        U = U.flatten().reshape(dim[0], dim[1], D)
        # note here that the A is not a skew-symetric matrix
        # so the number of independent params is half of the 
        # number of the elements in A
        # U, A are both tensor, torch.tensor
        
        return U, A, dim, Eig, error



    def contraction(self, T, gaugeU):
        # perform renormalization step by using projector U and U+
        # return T2

        U = gaugeU
        T1 = ncon([U,T,T,U],[[1,2,-3],[1,-4,5,6],[2,6,7,-9],[5,7,-8]],[7,5,6,1,2])
        # calc the coeff lambda for every renormalized tensor T 
        Lambda = T1.abs().max()
        T1 = T1/Lambda

        return T1, Lambda


    def rotate(self, T):
        # rotate the network for performing vertical coarse-graining
        # return T2
        return T.permute(self.Torder)




    def paramA(self, As):
        # parametrized As list to pure params list
        plist = []
        for k in range(len(As)):
            Ak = As[k]
            shape = Ak.shape
            for i in range(0, shape[0]-1):
                for j in range(i+1, shape[1]):
                    plist.append(Ak[i][j])
        
        return plist 




    def getPhy(self, T, lambda_i, lambdasum, step_i):
        # get the physical properties from the current tensor
        # here the step_i means how many HOTRG iterations has been made
        # F := free energy per site
        beta = 1/(self.kB*self.Temp)
        Dim = T.size()
        Tmat = T.flatten().reshape(Dim[0] * Dim[1], Dim[2] * Dim[3])
        Ttrace = Tmat.trace()
        # print('Trace of the current tensor: ', Ttrace)
        lambdasum += - 1/beta*(math.log(lambda_i)/2**step_i)
        f = - 1/beta*math.log(Ttrace)/2**step_i + lambdasum
        
        return f, lambdasum


    def construcUlist(self, Aflat: list):
        # Aflat : list containing all params
        Ulist = []
        count = 0
        for i in range(self.RGstep):
            shapeAi = self.Ashapes[i]
            typeAi = self.Atypes[i]
            dimTi = self.dim[i]

            Ai = torch.zeros(shapeAi, dtype=typeAi)
            for i in range( 0, shapeAi[0] - 1 ):
                for j in range( i+1, shapeAi[1] ):
                    Ai[i][j] = Aflat[count]
                    Ai[j][i] = - Aflat[count]
                    count += 1
            
            U = expm(Ai)

            Dcut = min(self.D, dimTi[2]*dimTi[3])
            U = U[:, :Dcut]
            U = U.flatten().reshape(dimTi[0], dimTi[1], Dcut)
            #print(U.requires_grad)
            Ulist.append(U)


        return Ulist

    



    def forward(self, Aflat):
        """
        in this forward function, the loss is calculated by inputing
        the PARAMETER A, which is a list containing gauges to perform TRG
        
        A ---> list[] 
        """
        
        T0 = self.T0
        Us = self.construcUlist(Aflat)
        fe = 0
        T_ = []
        U_ = []
        for i in range(self.RGstep):
            if i >= self.RGstep - 2:
                T_.append(T0)
                U_.append(gaugeU)
            gaugeU = Us[i]
            T1, Coef = self.contraction(T0, gaugeU)
            T0 = self.rotate(T1)
            fe -= self.Temp * torch.log(Coef) / 2**(i+1)


        return fe



    def go(self):
        """
        this is the main function to perform dTRG here
        """
        return None






if __name__ == '__main__':

    
    params = {
        'J': -1,
        'D': 3, 
        'kB': 1,
        'Temp': 2.5,    
        'RGstep': 30,
        'Torder': [3, 0, 1, 2],
        'dtype': torch.float64,
        'Nepochs':100
    }
    # device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    def closure():
        optimizer.zero_grad()
        loss = dtrg.forward(torparamA)
        # this backward() call calc the gradients of all params
        loss.backward()
        return loss


    err = []
    for i in range(len(rigtemp)):
        temp = rigtemp[i]
        fe_rig = rigfe[i]
        params['Temp'] = temp
        dtrg = dHOTRG(params)
        dtrg.initT()
        dtrg.initA()
        A = dtrg.A

        # paramA contains all independent params in RGsteps
        paramA = torch.tensor(dtrg.paramA(A))
        torparamA = torch.nn.Parameter(paramA)
        optimizer = torch.optim.LBFGS([torparamA], lr=1, max_iter=1, max_eval=None, tolerance_grad=1e-10, tolerance_change=1e-12, history_size=100, line_search_fn=None)
        
        

        for ei in range(dtrg.Nepochs):
            # this optimizer.step() call makes updates to the learnable params
            loss = optimizer.step(closure)
            fei = loss.item()
            print(' %d-th iteration, fe= %.15f '% (ei, fei))
            #steps.append(ei)
            #fe.append(fei)

        erri = abs((fei - fe_rig) / fe_rig)
        err.append(erri)
        print('已完成%d',i)

    err_np = np.array(err,dtype=np.float64)
    print(err_np)
    np.save('dtrg'+str(dtrg.D)+'.npy', err_np)
        
        
        


    # print('\n')
    # print("hotrg")
    # print('temp     :', temp)
    # print('D        :', dtrg.D)
    # print('RGsteps  :', dtrg.RGstep)
    # print('OPT iters:', dtrg.Nepochs)
    # print('result EXACT        :', ferig[i])
    # print('result HOTRG        :', res_hotrg)
    # print('result 1st Aforward :', res_Afwd.item())
    # print('result by optimized :', fei)




