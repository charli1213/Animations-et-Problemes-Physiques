# --- Importation des modules ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 0.4


# --- Paramètres physiques ---
xmax = 1.5 # Étendue maximale du domaine
nx   = 501
dx   = (2*xmax+1)/nx
x    = np.linspace(-xmax,xmax,nx)
nt   = 40000
dt   = 0.0001 # We're stuck here 0.01
tmax = nt*dx
times = np.linspace(0,tmax,nt)
n_out = 100 #25

# --- Paramètres particule
m    = 10. # Masse de la particule 
hbar = 1./(2*np.pi) # Constante de Plank
E_i  = 5.
k    = (2*m*E_i)**0.5/hbar


# --- Fonctions ---
def potentiel(x,p=3) :
    # Potentiel harmonique :
    omega = 5
    Vmax = 100
    xmax = 0.5
    if p==0 : return omega**2*(x**2) # harmonique
    if p==1 : return np.where((x<xmax)*(x>-xmax),np.zeros(len(x)),100) # Puit infini
    if p==2 : return 1*np.where((x<0),np.zeros(len(x)),1) # Bump
    if p==3 : return np.ones(len(x))*0.2
    
def initial_condition(x, x0 = -0.25) :
    # Construction de l'onde physique dans un potentiel zero :
    psi_0 = np.exp(1j*(k*(x-x0)))
    # Transformation en paquet d'ondes
    normal_func = np.exp(-( (x-x0)**2)/0.01)
    psi_0 = psi_0*normal_func
    # Normalisation :
    return (psi_0/np.linalg.norm(psi_0)).reshape(1,nx)


def hamiltonian_operator(x) :
    """ Création de l'opérateur matriciel associé à Schrodinger. """
    H = np.zeros((len(x),len(x)))
    for i,Vx in enumerate(potentiel(x)) :
        H[i,i] = hbar**2/(m*dx**2) + Vx
        if (i<len(x)-1): H[i,i+1] = -hbar**2/(2*m*dx**2)
        if i>0         : H[i,i-1] = -hbar**2/(2*m*dx**2)
    return H


def leapfrog_timestep(psi,H, nt=nt, dt=dt, n_out = n_out) :
    # Definitions  :
    psi_out  = np.zeros((int(nt/n_out),1,nx), dtype='complex')
    psi_i = psi_out[0] = psi # Temps t=0.
    # First Euler timestep :
    psi_half = psi - (1j/hbar)*(dt/2)*psi.dot(H.T)
    # Main loop : 
    for it in range(1,nt) :
        # Leapfrog RHS :
        psi_i    = psi_i   - dt*(1j/hbar)*H.dot(psi_half.T).T
        psi_half = psi_half- dt*(1j/hbar)*H.dot(psi_i.T).T
        # Renormalisation :
        psi_i /= np.linalg.norm(psi_i)
        # Enregistrement :
        if (it%n_out==0): psi_out[int(it/n_out)] = psi_i
    return psi_out
    

# --- Run ---
psi0 = initial_condition(x) 
H = hamiltonian_operator(x)
psi_out  = leapfrog_timestep(psi0,H)
P_out    = np.sqrt(np.real(psi_out*(psi_out.conj())))

# --- Animation ---
fig = plt.figure(figsize = (14,8))
plt.plot(x, potentiel(x)/70-0.2,c='k',linestyle = ':',label = r'$V(x)$')
imR,  = plt.plot(x,psi_out[0,0,:].real, c='b',label = r'$\mathrm{Re}\lbrace \Psi(x,t)\rbrace$')
imC,  = plt.plot(x,psi_out[0,0,:].imag, c='r',label = r'$\mathrm{Im}\lbrace \Psi(x,t)\rbrace$')
imP,  = plt.plot(x,P_out  [0,0,:],c='k',label = r'$|\Psi(x,t)|$')
plt.legend(loc = 'upper right')

def init() :
    imR.set_data(x,psi_out[0,0,:].real)
    imC.set_data(x,psi_out[0,0,:].imag)
    imP.set_data(x,P_out[0,0,:])
    return imR, imC, imP,

def animate(it) :
    imR.set_data(x,psi_out[it,0,:].real)
    imC.set_data(x,psi_out[it,0,:].imag)
    imP.set_data(x,P_out[it,0,:])
    return imR, imC, imP

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(nt/n_out), interval=50, repeat = True)
plt.ylabel(r'$|\Psi|$')
plt.xlabel(r'$x$')
plt.ylim(-0.5,0.5)
plt.xlim(-xmax,xmax)

#writervideo = animation.FFMpegWriter(fps=35)
#anim.save('schrodinger_harmonique.mp4', writer = writervideo)
plt.show()
        


    
