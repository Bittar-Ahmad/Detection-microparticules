# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:05:06 2021

@author: user
"""

import matplotlib.pylab as plt
import numpy as np
import numpy.random
import scipy
#import skimage
import pymrt as mrt
import raster_geometry as rg
#import pymrt.geometry



# constants
def setParameters(shapeimg=350,# size of the image
                  radius=80, # radius of HR ball
                  step=0.1, # relative to 1
                  slices=7,# number of 2D samples
                  zoomfactor=0.1,
                  approxorder=1):
    return(shapeimg,radius,step,slices,zoomfactor,approxorder)

# globals, avoid using them
shapeimg,radius,step,slices,zoomfactor,approxorder = setParameters()

# perfect sphere
ball1=rg.sphere(shapeimg,radius,position=[0.5,0.7,0.7]).astype(float)
ball1.shape

plt.figure()
plt.imshow(ball1[int(shapeimg/2),:,:])
plt.show()

## Random placement, then downsample
center=np.random.normal(loc=0.5,scale=step,size=3)

ball1=rg.sphere(shapeimg,radius,position=center).astype(float)
print("center=",center)

from scipy.ndimage.interpolation import zoom
smallball1=zoom(ball1,zoomfactor,order=1)
plt.figure()
plt.imshow(zoom(ball1[:,:,int(shapeimg*0.4)],zoomfactor,order=1))
plt.show()

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
#logging.debug('A debug message!')
#logging.info('We processed %d records', len(processed_records))

def sampleBall(debug=True):
    # parameters as local variables
    shapeimg,radius,step,slices,zoomfactor,approxorder = setParameters()
    
    n=slices
    slicelist = ((np.arange(n)-(n-1)/2)*(2*radius/n)+(shapeimg/2)).astype(int)
    logging.debug("slicelist=%s",str(slicelist))
    disklist = []
    centers = []
    for i in range(n):
        # create high resolution ball randomly shifted
        center=np.random.normal(loc=0.5,scale=step,size=3)
        logging.debug("center=%s",str(center))
        hrBall=rg.sphere(shapeimg,radius,position=center).astype(float)
        lrDisk=zoom(hrBall[slicelist[i],:,:],zoomfactor,order=approxorder)
        disklist.append(lrDisk)
        centers.append(center)
    return(centers,slicelist, np.array(disklist))

centers,slicelist,mysamples=sampleBall()
def imshown(imlist,cmap="gray"):
    w=10
    h=10
    fig=plt.figure(figsize=(9, 2))
    columns = len(imlist)
    rows = 1
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imlist[i-1],cmap=cmap)
        plt.axis('off')

    plt.show()
    
imshown(mysamples,cmap="gray")
##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
#################### Simulation and estimation ###############################
##############################################################################

# Parameters

# Dimension of the 3D image
Nx,Ny,Nz = (350,350,350)

# Radius and center of the sphere
Radius = 80
np.random.seed(42)
center=np.random.normal(loc=0.5,scale=step,size=3)

# Particle simulation
# perfect sphere
ball =rg.sphere((Nx,Ny,Nz),Radius,position=center).astype(float)
print(ball.shape)

# Zoom
# Zoom factor
zoomfactor = 0.1
nx = int(zoomfactor*Nx)
ny = int(zoomfactor*Ny)
nz = int(zoomfactor*Nz)
R = Radius*zoomfactor


from scipy.ndimage.interpolation import zoom

particle = zoom(ball,zoomfactor,order=1)

plt.figure()
plt.imshow(ball[:,:,int(Nz*center[2])])
plt.scatter([Nx*center[1]],[Ny*center[0]], marker="x")
plt.show()


plt.figure()
plt.imshow(particle[:,:,int(nz*center[2])])
plt.scatter([nx*center[1]],[ny*center[0]], marker="x")
plt.show()

# Diagonal of the frames
d = np.sqrt(nx**2+ny**2)

# z-coordinates of the frames
frames  = np.arange(0,nz)
# Number of frames
m = len(frames)
##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############# Total variation 3D matrix of the image frames ##################
##############################################################################

def W_mat(image, frames):
    nx,ny,nz = image.shape
    W = np.zeros((nx,ny,np.size(frames,0)))
    for i in range(nx):
        for j in range(ny):
            for k in range(np.size(frames,0)):
                varX = 0
                varY = 0
                if i==0:
                    varX += (image[i+1,j,k]-image[i,j,k])**2
                elif i==nx-1:
                    varX += (image[i-1,j,k]-image[i,j,k])**2
                else:
                    varX += (image[i-1,j,k]-image[i,j,k])**2+(image[i+1,j,k]-image[i,j,k])**2
                if j==0:
                    varY += (image[i,j+1,k]-image[i,j,k])**2
                elif j==ny-1:
                    varY += (image[i,j-1,k]-image[i,j,k])**2
                else:
                    varY += (image[i,j-1,k]-image[i,j,k])**2+(image[i,j+1,k]-image[i,j,k])**2
                W[i,j,k] = np.sqrt(varX+varY)
    return W

plt.figure(figsize=(15,10))
plt.imshow(W_mat(particle,frames)[:,:,int(nz*center[2])])
##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
##################### Non reformulated Objective function ####################
##############################################################################

def objective_function(image,frames):
    nx,ny,nz = image.shape
    m = len(frames)
    W = W_mat(image,frames)
    
    def obj(alpha,beta,xi,gamma,eta):
        res = 0
        for k in range(m):
            #print(k)
            for i in range(nx):
                for j in range(ny):
                    res += W[i,j,k]*(-2*i*alpha -2*j*beta + xi - eta[k] +i**2+j**2)**2
        return res/2
    return obj

obj = objective_function(particle,frames)
print(obj(10,10,10,10,np.ones(m)))
##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############## Creation of a function for matrices A and b ###################
##############################################################################

def Create_A_and_b(W):
    "Cette fonction permet de creer la matrice A de taille n * 4+m et le vecteur b de taille n"
    
    #Convertir la matrice W en 2D -- Taille: n * s(3) avec n = s(1)*s(2)
    s1 = np.size(W,0)
    s2 = np.size(W,1)
    s3 = np.size(W,2)
    
    W_2D = W.reshape(s1*s2, s3)

#Definir le vecteur f_1 de taille 4+m (vecteur de 0 sauf le 1er element est 1)    
    f_1 = np.zeros((4+s3,1))
    f_1[0] = 1   
    
#Definir le vecteur f_2 de taille 4+m (vecteur de 0 sauf le 2eme element est 1)
    f_2 = np.zeros((4+s3,1))
    f_2[1] = 1    
    
#Definir le vecteur f_3 de taille 4+m (vecteur de 0 sauf le 3eme element est 1)
    f_3 = np.zeros((4+s3,1))
    f_3[2] = 1    

#Definir le vecteur f_{4+i} de taille 4+m (vecteur de 0 sauf le (4+i)th element est 1)
    f_4_i = np.zeros((4+s3,1))   
    
# Definir les vecteurs x et y de taille n chacun
    x = np.zeros((s1*s2,1))
    y = np.zeros((s1*s2,1))
    for jj in range(s1*s2):
        x[jj] = np.floor((jj)/s1)
        y[jj] = np.mod(jj,s1)

        
#Definir un vecteur unite de taille n
    vecteur_unitaire = np.ones((s1*s2,1));    
        
    A = np.zeros((np.size(W_2D, 0), 4+s3, s3))
    b = np.zeros((np.size(W_2D, 0), s3));
    
    listA = [] #empty listA
    listb = []  #empty listb
    
    for kk in range(s3):
        f_4_i[4+kk] = 1
        A[:, :, kk] = np.diag(np.sqrt(W_2D[:, kk])) .dot(-2*np.kron(np.transpose(f_1), x) - 2*np.kron(np.transpose(f_2), y) + np.kron(np.transpose(f_3), vecteur_unitaire) - np.kron(np.transpose(f_4_i), vecteur_unitaire))
        listA.append(A[:, :, kk])
        
        b[:, kk] = np.multiply(-np.sqrt(W_2D[:, kk]), np.transpose(x**2)) + np.multiply(np.sqrt(W_2D[:, kk]), np.transpose(y**2))
        listb.append(b[:, kk])
        
          
   #Concatenation dans une grande matrice (c'est important pour CVX)           
    permuted_A = np.transpose(A, (0, 2, 1))
    matrice_A = permuted_A.reshape(np.size(W_2D,0)*s3, 4+s3)
    vecteur_b = b.reshape(np.size(b,0) * np.size(b,1))
    
    return matrice_A, vecteur_b, listA, listb

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############### Keeping only frames intersecting the sphere ##################
##############################################################################

def intersecting_frames(image,frames):
    start = -1
    end = -1
    index = 0
    m = len(frames)
    while end<0 and index<m:
        M = np.amax(image[:,:,frames[index]])
        if M>0 and start<0:
            start = index
        if start >= 0 and M==0:
            end = index
        index+=1
    if end<0:
        end = m
    return frames[start:end]

print(intersecting_frames(particle, frames))
print(np.amax(particle[:,:,11]))
print(np.amax(particle[:,:,12]))


framesInter = intersecting_frames(particle, frames)
mInter = len(framesInter)
print(mInter)

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############### Solve the non reformulated objective with CVX ################
##############################################################################

import cvxpy as cp

# Variables
alpha = cp.Variable(1)
beta = cp.Variable(1)
gamma = cp.Variable(1)
xi = cp.Variable(1)
eta = cp.Variable(mInter)

# Objective
obj = objective_function(particle,framesInter)
objective = cp.Minimize(obj(alpha,beta,xi,gamma,eta))

# Constraints
constraints = [alpha**2+beta**2-xi <= 0, xi - d**2 <= 0, -eta <= 0, eta+(gamma-framesInter)**2 - R**2 <= 0]

# Problem
prob = cp.Problem(objective,constraints)

# Solving
prob.solve(solver = cp.ECOS, verbose = True, max_iters=100)

print("Optimal value", prob.value)
print("Status :", prob.status)

print("alpha = ", alpha.value) 
print("beta = ", beta.value) 
print("gamma = ", gamma.value) 

print("Real center: ", center*nx)

Z = int(nz*center[2])

#for i in range(-8,9,2):
#    plt.figure(figsize=(20,10))
#    plt.imshow(particle[:,:,Z+i])
#    plt.scatter([nx*center[1]],[ny*center[0]], marker="x")
#    plt.scatter([beta.value],[alpha.value], marker="x", color='red')
#    plt.show()

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
################# Solve the reformulated objective with CVX ##################
##############################################################################

def objective_function_reformulated_CVX(image,frames):
    import cvxpy as cp
    
    # Variables
    alpha = cp.Variable(1)
    beta = cp.Variable(1)
    gamma = cp.Variable(1)
    xi = cp.Variable(1)
    eta = cp.Variable(mInter)
    m = len(frames)
    W = W_mat(image,frames)
    A,b,listA,listb = Create_A_and_b(W)
    
    cost = cp.sum_squares(A @ cp.hstack([alpha,beta,xi,gamma,eta]) - b)/2
    
    # Constraints
    constraints = [alpha**2+beta**2-xi <= 0, xi - d**2 <= 0, -eta <= 0, eta+(gamma-framesInter)**2 - R**2 <= 0]
    
    prob = cp.Problem(cp.Minimize(cost), constraints)
    
    prob.solve(solver = cp.ECOS, verbose = True, max_iters=1000)
    
    print("Optimal value", prob.value)
    print("Status :", prob.status)

    print("alpha = ", alpha.value) 
    print("beta = ", -beta.value) 
    print("gamma = ", gamma.value) 

    print("Real center: ", center*nx)

    Z = int(nz*center[2])

    #for i in range(-8,9,2):
    #    plt.figure(figsize=(20,10))
    #    plt.imshow(particle[:,:,Z+i])
    #    plt.scatter([nx*center[1]],[ny*center[0]], marker="x")
    #    plt.scatter([-beta.value],[alpha.value], marker="x", color='red')
    #    plt.show()
        
objective_function_reformulated_CVX(particle,framesInter) 

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###

###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###

###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###

########################### METHODE DE NEWTON ################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###

###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###

###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############## The function without barrier and constraints ##################
##############################################################################

def fct(listA, listb):
    
    " Cette fonction minimise (1/2) sum_{i=1}^m [ || A_i*v - b_i ||^2 ] "
    
    m = len(listA) # m should be equal to the total intersected frames
    def f(v): 
        return np.sum([np.linalg.norm(listA[i]@v - listb[i])**2 for i in range(m)])/2
    return f

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


###############################################################################
################## The function with barrier and constraints ##################
##############################################################################

def fct_bar(listA, listb, z, R, d, t): # t est le paramètre de barrière
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    " Our original minimization without the barrier: "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    "minimize { (1/2) sum_{i=1}^m [ || A_i*v - b_i ||^2 ] }"
    " s.t. -eta_i <=0 "
    "      eta_i + (gamma - z_i)^2 - R^2 <=0 "
    "      alpha^2 + beta^2 - psi <=0"
    "      psi - d^2 <=0"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """        """
    """"""""""""""
    """        """
    
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    " Our minimization with the barrier: "
    """"""""""""""""""""""""""""""""""""""""""""""""""""""
    " minimize { (1/2) sum_{i=1}^m [ || A_i*v - b_i ||^2 ]  - (1/t) * sum_{k=1}^4 [ ln(-c_k(v)) ] }"
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
    """        """
    """"""""""""""
    """        """
    
    "Define v = [alpha beta psi gamma eta_1 ... eta_m]^T : a vector of size 4+m"
    " v[0] = alpha "
    " v[1] = beta "
    " v[2] = psi "
    " v[3] = gamma "
    " v[4:] = eta_1 ... eta_i "
    
    """        """
    """"""""""""""
    """        """
    
    m = len(listA) # m should be equal to the total intersected frames
    def f(v):
        constr1 = v[4:] # constr1 = -c_1(v) = eta_i, with i=1,...m
        
        constr2 = -(v[4:] + (v[3]-z)**2 - R**2) # constr2 = -c_2(v) = -(eta_i + (gamma - z_i)^2 - R^2)
        constr3 = [-(v[0]**2 + v[1]**2 - v[2])] # constr3 = -c_3(v) = -(alpha^2 + beta^2 - psi)
        constr4 = [-(v[2] - d**2)] # constr4 = -c_4(v) = -(psi - d^2)
        
        constr = np.concatenate((constr1,constr2,constr3,constr4)) #constr = [constr1 constr2 constr3 contsr4]
        
        if np.min(constr) <= 0:
            #raise NameError("v non admissible")
            return np.inf
        else:
            "return f(v) = (1/2) sum_{i=1}^m [ || A_i*v - b_i ||^2 ]  - (1/t) * sum_{k=1}^4 [ ln(-c_k(v)) ] "
            return np.sum([np.linalg.norm(listA[i]@v - listb[i])**2 for i in range(m)])/2 - (1/t)*np.sum(np.log(constr))
    return f

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
################ Gradient en v de la fonction à minimiser ####################
##############################################################################

def grad_f(listA, listb, z, R, d, t, v):
    
    "Define v = [alpha beta psi gamma eta_1 ... eta_m]^T : a vector of size 4+m"
    " v[0] = alpha "
    " v[1] = beta "
    " v[2] = psi "
    " v[3] = gamma "
    " v[4:] = eta_1 ... eta_i "
    
    "Voir sur overleaf pour la formule obtenue du gradient"
    
    m = len(listA) # m should be equal to the total intersected frames
    n = len(v) # n should be equal to (4 + m)
    
    "res = sum_{i=1}^m [ A_i' * ( A_i*v - b_i ) ] "
    res = np.sum([listA[i].T@(listA[i]@v - listb[i]) for i in range(m)])
    
    bar = np.zeros(n)
    
    " terme dû à la fonction barrière pour constr3 et constr4 "
    x = (-(v[0]**2 + v[1]**2 - v[2])) # x = -(alpha^2 + beta^2 - psi)
    bar[0] += -2*v[0]/x # (-2 * alpha) / x
    bar[1] += -2*v[1]/x # (-2 * beta) / x
    bar[2] += 1/x - 1/(d**2-v[2]) # 1/x - 1/(d^2 - psi)
    
    for j in range(4,n): # j goes from 4 to (4 + m)
        u = v[j] # u = eta_i , with i belongs to [1, m]
        
        " terme dû à la fonction barrière pour constr1 et constr2"
        x = R**2 - u - (v[3]-z[j-4])**2 # x = R^2 - eta_i - (gamma - z_i)^2 , with i in [1, m]
        bar[3] += -2*(v[3]-z[j-4])/x # -2(gamma - z_i) / x , with i in [1, m]
        bar[j] += 1/u - 1/x # 1/eta_i - 1/x  , with i in [1, m]
    return res - bar/t

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
###################### Calcul de la Hessienne ################################
##############################################################################

def hess_f(listA, listb, z, R, d, t, v):
    
    "Define v = [alpha beta psi gamma eta_1 ... eta_m]^T : a vector of size 4+m"
    " v[0] = alpha "
    " v[1] = beta "
    " v[2] = psi "
    " v[3] = gamma "
    " v[4:] = eta_1 ... eta_i "
    
    "Voir sur overleaf pour la formule obtenue de la Hessienne"
    
    m = len(listA) # m should be equal to the total intersected frames
    n = len(v) # n should be equal to (4 + m)
    
    res = np.sum([listA[i].T@listA[i] for i in range(m)]) # res = sum_{i=1}^m [ A_i^T * A_i ]
    bar = np.zeros((n,n))
    
    x = (v[0]**2 + v[1]**2 - v[2])**2
    bar[0,0] += (-2*(v[2] - v[0]**2 - v[1]**2) - 4*v[0]**2)/x
    
    bar[1,1] += (-2*(v[2] - v[0]**2 - v[1]**2) - 4*v[1]**2)/x
    
    bar[2,2] += -1/x - 1/(v[2]-d**2)**2
    
    b = -4*v[0]*v[1]/x
    bar[0,1] += b
    bar[1,0] += b
    b = 2*v[0]/x
    bar[0,2] += b
    bar[2,0] += b
    b = 2*v[1]/x
    bar[1,2] += b
    bar[2,1] += b
    for j in range(4,n):
        u = v[j]
        # terme dû à la fonction barrière pour constr1 et constr2
        x = (R**2 - u - (v[3]-z[j-4])**2)**2
        bar[3,3] += (-2*(R**2 - u - (v[3]-z[j-4])**2) - 4*(v[3]-z[j-4])**2)/x
        bar[j,j] += -1/u**2 - 1/x
        b = 2*(v[3]-z[j-4])/x
        bar[3,j] += b
        bar[j,3] += b
    return  res - bar/t

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
################## Backtracking line search ##################################
##############################################################################

def lineSearch(f,x,delta,grad,alpha,beta,t=1): # 0<alpha<1/2 et 0<beta<1
    while f(x+t*delta) >= f(x) + alpha*t*(grad.T@delta) :
        t = beta*t
        #print(t) # on reste toujours dans la boucle while, on ne sort jamais!
        #if (t<0.000000000000000001):
         #   break
    return t

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
######################## Centering step ######################################
##############################################################################

# On retourne la liste des points et des decr pour les graphes à la fin
def centering_step(listA, listb, z, R, d, t, v0, crit, alpha, beta): # t et eps > 0
    # méthode de Newton
    res = [v0]
    v = v0
    decr = np.inf
    liste_decr=[]
    f = fct_bar(listA, listb, z, R, d, t)
    grad = grad_f(listA, listb, z, R, d, t, v)
    hess = hess_f(listA, listb, z, R, d, t, v)
    hess_inv = np.linalg.inv(hess)
    delta = -hess_inv@grad
    decr = grad.T@hess_inv@grad
    
    while decr/2 >= crit:
        v = v + lineSearch(f,v,delta,grad,alpha,beta)*delta  # Il bloque ici
        res.append(v)
        grad = grad_f(listA, listb, z, R, d, t, v)
        hess = hess_f(listA, listb, z, R, d, t, v)
        hess_inv = np.linalg.inv(hess)
        delta = -hess_inv@grad
        decr = grad.T@hess_inv@grad
        liste_decr.append(decr)
        #print("decr = ",decr)
        print(delta)
        print("v_cent = ",v)
        #print("f_bar(v_cent) = ",f(v))
    return (res,liste_decr)

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
############################# Barrier method #################################
##############################################################################

# On retourne la liste des points et des decr pour les graphes à la fin
def barr_method(listA, listb, z, R, d, v0, eps, crit, alpha, beta, t0, mu): # t0, mu et eps > 0
    res = [v0]
    liste_decr = []
    n = len(v0)
    v = v0
    t = t0
    while n/t >= eps:
        print("v = ",v)
        print("f(v) = ",fct(listA,listb)(v))
        (v, liste_decr_centering) = centering_step(listA, listb, z, R, d, t, v,crit, alpha, beta)
        liste_decr+= liste_decr_centering
        t = mu*t
        res = res + v
        v = v[-1]
    return (res,liste_decr)

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
###################### Test avec les matrices A et b #########################
##############################################################################

W = W_mat(particle,framesInter)
A, b, listA, listb = Create_A_and_b(W)

mInter = len(framesInter)
v0 = np.ones(4 + mInter)
v0[2] = 100

eps=0.001
crit=100
alpha=0.3
beta=0.4999

#print(barr_method(listA, listb, framesInter, R, d, v0, eps, crit, alpha, beta, t0=1, mu=10)[0][-1])

##############################################################################
##############################################################################
##############################################################################


###            ###                                          ###            ###                                       
##################                                          ################## 
###            ###                                          ###            ###


##############################################################################
################################ Random Test #################################
##############################################################################

n = 8
listA = [np.eye(n)]

listA[0][2,2]=0
listA[0][0,0]=0
listA[0][1,1]=0
b = np.ones(n)*10
b[2]=0

listb = [b]
R=50
d=60
z = np.zeros(n-4)
v0 = np.ones(n)*5
v0[3]=150
#v0 = np.random.rand(n)
#v0 = np.zeros(n)
#v0[3]=150

eps=0.001
crit=1
alpha = 0.3
beta = 0.99 

#print(barr_method(listA, listb, z, R, d, v0,eps,crit, alpha, beta, t0=1,mu=10)[0][-1])

###############################################################################
###############################################################################
###############################################################################




#A jeter apres...

############### Solve the reformulated objective with CVX ####################
#import cvxpy as cp

# Variables
#alpha = cp.Variable(1)
#beta = cp.Variable(1)
#gamma = cp.Variable(1)
#xi = cp.Variable(1)
#eta = cp.Variable((mInter,1))

# Objective
#obj_reformulated = objective_function_reformulated(particle,framesInter)
#objective_reformulated = cp.Minimize(obj_reformulated(alpha,beta,xi,gamma,eta))

# Constraints
#constraints = [alpha**2+beta**2-xi <= 0, xi - d**2 <= 0, -eta <= 0, eta+(gamma-framesInter)**2 - R**2 <= 0]

# Problem
#prob_reformulated = cp.Problem(objective_reformulated,constraints)

# Solving
#prob_reformulated.solve(solver = cp.ECOS, verbose = True, max_iters=100)

#print("Optimal value", prob_reformulated.value)
#print("Status :", prob_reformulated.status)

#print("alpha = ", alpha.value) 
#print("beta = ", beta.value) 
#print("gamma = ", gamma.value) 

#print("Real center: ", center*nx)

#Z = int(nz*center[2])

#for i in range(-8,9,2):
#    plt.figure(figsize=(20,10))
#    plt.imshow(particle[:,:,Z+i])
#    plt.scatter([nx*center[1]],[ny*center[0]], marker="x")
#    plt.scatter([beta.value],[alpha.value], marker="x", color='red')
#    plt.show()
##############################################################################