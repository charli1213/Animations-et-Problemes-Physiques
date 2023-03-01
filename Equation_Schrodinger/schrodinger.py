# === Imporation des modules :
import numpy as np
from scipy.sparse import linalg
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 0.4

# --- Paramètres physiques :
m    = 1. # Masse de la particule
hbar = 1. # Constante de Plank
xmax = 20 # Étendue maximale du domaine
nx   = 301
dx   = (2*xmax+1)/nx
x    = np.linspace(-xmax,xmax,nx)
nt   = 400
tmax = 400
times = np.linspace(0,tmax,nt)
dt   = tmax/nt
Vpot = 1

# --- Fonctions : p
def potential(x) :
    # Potentiel harmonique :
    omega = 0.08
    if Vpot==0 : return 0.5*m*omega**2*x**2
    # Puit infini :
    if Vpot==1 : return np.where((x<10)*(x>-10),np.zeros(len(x)),1000)

def hamiltonian_operator(x) :
    H = np.zeros((len(x),len(x)))
    for i,Vx in enumerate(potential(x)) :
        H[i,i] = hbar**2/(m*dx**2) + Vx
        if (i<len(x)-1): H[i,i+1] = -hbar**2/(2*m*dx**2)
        if i>0         : H[i,i-1] = -hbar**2/(2*m*dx**2)
    return H

def solution_temporelle(t_vec,E_i) :
    times    = t_vec.reshape(1,len(t_vec))
    energies = E_i.reshape(len(E_i),1)
    argument = energies.dot(times)
    return np.exp(-1j*argument/hbar)
    
    
# ==== calculs ====
# --- Solution de l'opérateur linéaire. 
istate = [1,3,4]
H           = hamiltonian_operator(x)
E_i, psi_ix = linalg.eigs(H,k=25, which='SR')
phi_it      = solution_temporelle(times,E_i)
### À noter : scipy.linalg.eig() 'bug' vraiment beaucoup.


# --- Séparation (fusion) des variables et normalisation.
psi = np.array([psi_ix[:,i:i+1].dot(phi_it[i:i+1,:]) for i,E in enumerate(E_i)]) # Choix des états
psi = psi[istate].sum(axis=0) # Addition des états choisis.
psi = psi/np.linalg.norm(psi[:,0]) # normalisation
Prob = np.sqrt((psi*psi.conj())).real # Probabilité
psi2 = (psi*psi.conj()).real
Xmean = psi2.T.dot(x.reshape(nx,1))



if __name__ == "__main__"  :
    # ==== Animation ====
    # --- Création de la figure 
    fig = plt.figure(figsize = (7,5))
    plt.plot(x, potential(x)-0.1,label = r'$V(x)$'  ,c='k',linestyle = ':')
    xvline = plt.axvline(Xmean[0],c='k',linestyle = '-.')
    xvtext = plt.text(x[0],-0.15,r"$\langle\Psi|x|\Psi\rangle$", zorder = 4)
    imP, = plt.plot(x,Prob[:,0],label = r'$|\Psi(x,t)|$',c='r')
    imR, = plt.plot(x,psi.real[:,0],label = r'$\mathrm{Re}\lbrace \Psi(x,t)\rbrace$'  ,c='b')
    im_text = plt.text(10,-0.255,"Temps = 0 s", zorder = 4)
    plt.ylim(-0.3,0.3)
    plt.xlim(-20,20)

    # --- Fonctions d'animation
    def init() :
        imR.set_data(x,psi.real[:,0])
        imP.set_data(x,Prob[:,0])
        im_text.set_text("Temps = {} s".format(0))
        xvline.set_xdata(Xmean[0])
        xvtext.set_position((Xmean[0],-0.15))
        return imR, imP, im_text, xvline

    def animate(it) :
        imR.set_data(x,psi.real[:,it])
        imP.set_data(x,Prob[:,it])
        im_text.set_text("Temps = {} s".format(it*dt))
        xvline.set_xdata(Xmean[it])
        xvtext.set_position((Xmean[it]+1,-0.15))
        return imR, imP, im_text, xvline

    # --- Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=nt, interval=50, repeat = True)
    plt.legend(loc = 'upper right')
    plt.xlabel('x')
    plt.ylabel(r'$|\Psi|$')
    str_states = ','.join(str(i) for i in istate)
    plt.title(r'$\Psi (x,t)$ for $E_{{{}}}$'.format(str_states))
    fig.show()



