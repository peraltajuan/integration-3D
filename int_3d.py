import numpy as np
from scipy.special import sph_harm
from scipy.special import assoc_laguerre
from math import factorial
from  numpy.polynomial.chebyshev import  chebgauss 


def Rad_func(n,l,r):
    ''' 
    This function returns the normalized radial hydrogenic funcion R_nl for a given r (or array of r)
    '''
    L = assoc_laguerre( ''' Complete '''  )
    Norm = ''' Complete ''' 
    R =  ''' Complete '''
    return R


def xyz2ang(x,y,z):
    '''
    convert xyz to spherical coordinates theta and phi
    '''
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = ''' Complete '''
    phi   = ''' Complete '''
    # clean possible nans
    phi = np.nan_to_num(theta)
    theta   = np.nan_to_num(phi)
    return theta, phi

def radial_grid(n):
    '''
    calculate radial quadrature weights and points
    '''
    rmax = 1.00
    T, W = chebgauss(n)
    R  = ''' Complete '''
    W =  ''' Complete '''   # the weight includes the R**2 needed for integration
    return W,R

def fz(xyz,alpha):
    r2 = ''' Complete '''
    return xyz.T[2] * np.exp( -r2*alpha )

def main():
    # Read the data file with Levedev info: x y z weight
    N = 110
    x,y,z,w = np.loadtxt(f'Lebedev.{N}').T
    theta, phi = xyz2ang(x,y,z)

    # choose a specific combination of quantum numbers
   

    n = 3
    l = 2
    m = 0


    # check that the angular integral works

    sph = 4*np.pi* np.abs( sph_harm(m, l, theta, phi, out=None) )**2
    I = w @ sph
    print(f'integral for Y l={l:2g} m={m:2g}         =  {I:19.8f}')

    # now generate the quadrature for the radial part

    W, R = radial_grid(40)

    # obtain the radial function

    Rad = Rad_func(n,l,R)**2  
    I = W @ Rad  
    print(f'integral for R n={n:2g} l={l:2g}         =  {I:19.8f}')

    # test integral of r**2 = <r**2> and compare with analytical value

    Rad2 = ''' Complete '''
    I2 = W @ Rad2  
    print(f'Numerical  <r**2> n={n:2g} l={l:2g}      =  {I2:19.8f}')
    r2 =  0.5*n**2 *(5*n**2 + 1 -3*l*(l+1) )
    print(f'Analytical <r**2> n={n:2g} l={l:2g}      =  {r2:19.8f}')
    

#   "pz1" function = z*Exp(-0.82454724*r**2)
#   "pz2" function = z*Exp(-5.447178  *r**2)

#    create an array with all the xyz values on the grid using R (radial points) and theta, phi (angles) 


    AX = ''' Complete'''
    AY = ''' Complete''' 
    AZ = ''' Complete '''
    XYZ =np.array([AX, AY, AZ]).T 



#    create an array with all the weight products for each grid point

    W3D = np.outer(W,w).flatten()


#    create an array with the function values on the grid points

    F1 = fz(XYZ,0.82454724)
    F2 = fz(XYZ,5.447178)

#   Normalize 

    N1 = ''' Complete '''
    F1 = N1*F1
    N2 = ''' Complete '''
    F2 = N2*F2

#   Integrate 

    Ov = W3D @  (F1*F2) 
    print(f'Overlap numerical    <F1|F2>     =  {Ov:19.8f}')
    print(f'Overlap analytical   <F1|F2>     =  {0.375484:19.8f}')

if __name__=='__main__':
    main()




