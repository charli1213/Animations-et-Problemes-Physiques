import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle

nx, ny = shape = 128, 128
dx = dy = 1.
alpha = 1.
dt = 0.2
cfl = dx/dt
hot_element = 250
cold_element = -2
print('CFL condition = ',cfl)
T = np.zeros(shape)-10

# --- Fonctions importantes : 
def apply_init(T) :
    """ Application des conditions frontière (boundary conditions)."""
    T[0,:] = T[-1,:] = 0.
    T[:,0] = T[:,-1] = 0.
    T[20:40,20:40] = 250.
    T[78:98,78:98] = -2.
    return T


def first_derivative(T,dx=dx,dy=dy) :
    """ Calculate first derivative of a fields (T in that specific case)."""
    ddx = (np.roll(T,-1,0) - T)/dx
    ddy = (np.roll(T,-1,1) - T)/dy
    return ddx,ddy


def second_derivative(ddx,ddy,dx=dx,dy=dy) :
    """ 
    Calculate second derivative of a fields (T in that specific case).
    (also called Laplacian).
    """
    dd2x = (ddx - np.roll(ddx,1,0))/dx
    dd2y = (ddy - np.roll(ddy,1,1))/dy
    return dd2x,dd2y

    
def timestep(T, it, dt = dt, alpha = alpha) :
    """ Aplication du RHS """
    ddx, ddy  = first_derivative(T)
    dd2x,dd2y = second_derivative(ddx,ddy)
    return T + dt*alpha*(dd2x + dd2y) # Eq diffusion.


# === Calculs : 
nt = 15000
dt_fig = 10
T_r = np.zeros((int(nt/dt_fig),nx,ny))
for it in range(nt) :
    T = timestep(T,it)
    T = apply_init(T)
    if (it%dt_fig == 0) :
        T_r[int(it/dt_fig)] = T


# === Animation :
# --- Settings de la figure pré-animation : 
fig = plt.figure(figsize = (7,5))
axe = plt.axes()
im = axe.imshow(T_r[0],
                cmap = 'magma')
im_text = plt.text(85,10,"Temps = 0 s", c='white', zorder = 4)

im_box1 = axe.add_patch(Rectangle((20, 20), 19, 19, fill = False, edgecolor = 'red',zorder=12,lw=0.7,alpha=0.4))
im_box2 = axe.add_patch(Rectangle((78, 78), 20, 20, fill = False, edgecolor = 'blue',zorder=12,lw=0.7,alpha=0.7))

texbox1 = plt.text(29,29,"Élément\n chauffant\n ({}$\degree$C)".format(hot_element),
                   c='red',
                   alpha = 0.4,
                   fontsize = 'x-small',
                   ha='center',va='center')
texbox2 = plt.text(88,88,"Élément\n tiede\n ({}$\degree$C)".format(cold_element),
                   c='blue',
                   alpha = 0.8,
                   fontsize = 'x-small',
                   ha='center',va='center')


cb = plt.colorbar(im,ax=axe)

# --- Fine-tunning :
axe.set_title(r'Équation de diffusion $[\partial T/ \partial t =\alpha \nabla^2 (T)]$')
cb.set_label('Température [$\degree$C]')
fig.tight_layout()


# --- Fonctions d'animation : 
def init() :
    """ 
    Creation d'une image initale. Retourne un iterable, par
    convention.
    """
    im.set_data(T_r[0])
    im_text.set_text("Temps = {} s".format(0))
    
    return im,im_text,im_box1,im_box2,texbox1,texbox2

def animate(i) :
    """
    Creation d'une fonction iterable pour l'animation.
    """ 
    im.set_data(T_r[i])
    im_text.set_text("Temps = {} s".format(i*dt_fig))
    return im,im_text,im_box1,im_box2,texbox1,texbox2


# --- Enclenchement animation : 
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=1500, interval=10, blit=True)
plt.show()
