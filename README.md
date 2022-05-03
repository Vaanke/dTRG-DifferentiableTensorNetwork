# Description

The main program in this folder is
dtrg.py (Differentiable Tensor Network Algorithm) and
dTRGB.py (differentiable Tensor network algorithm considering The Last Environment)
They are all applied to the 2DIsing Model to calculate the free energy.

- eig.py : Use power method to calculate the largest eigenvalue of the matrix
- expm.py : Calculate e^A, where A is a matrix
- rig.py : onsager strict solution for some temperature points of 2DIsing Model
- TorNcon.py : A tool library for computing tensor shrinkage
- AsymLogm.py : get the antisymmetric matrix A from the unitary matrix U (norm), e^A = U

**Running dtrg or dtrgB directly should be able to run through, and can be behind these two files
Modify main program parameters, optimization times, renormalization times, temperature points and other parameters.**

If the program does not work or have other problems, please contact us by email
derricfan@gmail.com


