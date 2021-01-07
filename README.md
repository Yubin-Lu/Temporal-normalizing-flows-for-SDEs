# Temporal-normalizing-flows-for-SDEs
Try to approximate the solution of Fokker-Planck equation using sample paths of stochastic differential equations.  
I have tried to run two example using temporal normalizing flows code.  

Example 1  
We consider a 1-D double-well system with Brownian motion  
dX_t = f(X_t)dt + dB_t, where f(x)=4x-x^3.  
See TNFforDWwithBM.py  


Example 2  
I also tried to approximate the data of 'IceCore Oxygen18' , but I do not know how to justify the result.   
See IceCore.py    
In this code, I preprocess the data of 'IceCore Oxygen18' using this formula  
x_normal = 100 * (x-mean(x)) / mean(x)  

Example 3
Combining RealNVP with temporal normalizing flows to approximate a solution of 2-d Fokker-Planck equation, but this method still need to be improved.
See the fold 'TNFwithRealNVP'

Part of the code is from the author of the following article and GitHub repository

[1] Gert-Jan Both, Remy Kusters. Temporal normalizing flows. arXiv preprint arXiv:1912.09092v1, 2019.

[2] tonyduan. https://github.com/tonyduan/normalizing-flows

