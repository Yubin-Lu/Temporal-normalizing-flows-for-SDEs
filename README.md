# Temporal-normalizing-flows-for-SDEs
Try to approximate the solution of Fokker-Planck equation  
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
