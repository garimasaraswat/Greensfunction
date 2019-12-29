import numpy as np 

k_B=1.38064852e-23 #(m^2kg)/(s^2K)
e_charge=1.6021765e-19 #C
k_BeV=k_B/e_charge
hbar=1.0545718e-34 #J.s
m_e=9.1093837015e-31 #Kg
# Pauli spin matrices
sigma0 = np.array([[1,0],[0,1]],dtype=np.complex64)
sigmax = np.array([[0,1],[1,0]],dtype=np.complex64)
sigmay = np.array([[0,np.complex(0,-1)],[np.complex(0,1),0]],dtype=np.complex64)
sigmaz = np.array([[1,0],[0,-1]],dtype=np.complex64)
