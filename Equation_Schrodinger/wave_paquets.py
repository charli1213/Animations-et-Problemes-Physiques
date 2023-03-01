import numpy as np
import matplotlib.pyplot as plt




# --- Paramètres physiques :
m    = 1. # Masse de la particule
hbar = 1. # Constante de Plank
xmax = 100 # Étendue maximale du domaine
nx   = 501
dx   = (2*xmax+1)/nx
x    = np.linspace(-xmax,xmax,nx)
nt   = 10000
dt   = 0.001 # We're stuck here 0.01
tmax = nt*dx
times = np.linspace(0,tmax,nt)





# --- Fonctions : 
def initial_condition(x) :
    omega = 1
    beta = (m*omega/hbar)
    psi_0 =  ((beta/np.pi)**(0.25) * np.exp(-beta*(x-200*dx)**2/2)).reshape(nx,1)
    return psi_0/np.linalg.norm(psi_0)


def potential(x) :
    # Potentiel harmonique :
    #omega = 0.01
    #return 0.5*m*omega**2*x**2
    # Puit infini :
    xmax = 80
    return 10000000*np.where((x<xmax)*(x>-xmax),np.zeros(len(x)),1000)


def hamiltonian_operator(x) :
    H = np.zeros((len(x),len(x)))
    for i,Vx in enumerate(potential(x)) :
        H[i,i] = hbar**2/(m*dx**2) + Vx
        if (i<len(x)-1): H[i,i+1] = -hbar**2/(2*m*dx**2)
        if i>0         : H[i,i-1] = -hbar**2/(2*m*dx**2)
    return H


def leapfrog_timestep(psi,H, nt=nt, dt=dt) :
    # First euler timestep :
    psi_out = np.zeros((nt,nx,1))
    psi_i = psi_out[0] = psi
    psi_half = psi - (1j/hbar)*(dt/2)*H.dot(psi)
    
    # Main loop : 
    for it in range(1,nt) : 
        psi_i    = psi_i   - (1j/hbar)*dt*H.dot(psi_half)
        psi_half = psi_half- (1j/hbar)*dt*H.dot(psi_i   )
        # Enregistrement : 
        psi_out[it] = psi_i
    return psi_out
    
# --- Run
psi0 = initial_condition(x)
H = hamiltonian_operator(x)

psi_out = leapfrog_timestep(psi0,H)
psi_out = psi_out.reshape(nt,nx).transpose()
plt.plot(psi_out[:,::10])
plt.show()
        
