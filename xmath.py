# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:23:50 2016

@author: bstone
"""

import numpy as np

#math

def fracpart(x):
    """Fractional part of x.
    
    >>> fracpart(np.linspace(0,np.pi,num=4))
    array([ 0. , 0.0471..., 0.0943..., 0.141...]) # doctest: +NORMALIZE_WHITESPACE
    """
    return x - np.floor(x)

def rfracpart(x):
    """1 minus the fractional part of x.
    
    >>> rfracpart(np.linspace(0,np.pi,num=4))     
    array([ 1.,  0.952...,  0.905...,  0.858...])  # doctest: +NORMALIZE_WHITESPACE  
    """
    return 1-fracpart(x) 

def xlogx(x):
    """Returns a continuous version of x log abs(x). Explicitly:
    x log x when x > 0
    0 when x = 0
    x log(-x) when x < 0
    
    >>> xlogx(np.linspace(-2,2,num=9)) # doctest: +NORMALIZE_WHITESPACE
    array([-1.386..., -0.608..., -0. , 0.346..., 0. ,
           -0.346..., 0. , 0.608..., 1.386...])
    """
    result = x * np.log(np.abs(x))
    return np.where(x == 0, 0, result)
    
def angle(z, scale = 1):
    """Same as np.angle except for special handling 
    when z is 0 or not finite, and parameter for scaling the result.
    
    >>> angle(np.array([np.nan, 0, 1, 1j, -1, -1j,
                        np.inf, np.inf*1j, -np.inf, -np.inf*1j]), 
              scale=180/np.pi)
    array([nan, nan, 0., 90., 180., -90., inf, inf, inf, inf]) # doctest: +NORMALIZE_WHITESPACE
    """
    zeroind = (z == 0)
    result = np.where(np.isinf(z), np.inf, np.angle(z))
    result[zeroind] = np.nan
    return result*scale
    
def disk_to_plane(y, a=1):
    """Transforms complex y on the unit disk abs(y) < 1 to the 
    complex plane. Note that if abs(y) >= 1, some sort of infinity
    or nan will be returned.
    
    Args:
        y: An array of complex numbers.
        a: Adjusts how the disk is transformed to the plane. Default is
           a = 1, the Poincare disk method. a = 0 corresponds to the 
           Klein disk method. 
    Returns:    
        An array of complex numbers.
        
    >>> disk_to_plane(np.linspace(-1j,1j,num=9)) 
    ... # doctest: +NORMALIZE_WHITESPACE
    array([ nan - infj,   0. - 3.428...j,   0. - 1.333...j,
         0. - 0.533...j,   0. + 0.j        ,   0. + 0.533...j,
         0. + 1.333...j,   0. + 3.428...j,  nan + infj])    
    """
    norm = np.abs(y)**2
    top = a + np.sqrt(1+(a**2-1)*norm)
    return np.where(norm > 1, np.nan, y*top/(1-norm) )

def plane_to_disk(x, a=1):
    """Transforms complex x to the unit disk abs(y) < 1.
    
    Args:
        x: An array of complex numbers.
        a: Adjusts how the disk is transformed to the plane. Default is
           a = 1, the Poincare disk method. a = 0 corresponds to the 
           Klein disk method. 
    Returns:    
        An array of complex numbers y, with abs(y) < 1.
    
    >>> plane_to_disk(np.linspace(-2j,2j,num=9)) # doctest: +NORMALIZE_WHITESPACE
    array([-0.-0.618...j, -0.-0.535...j, -0.-0.414...j, -0.-0.236...j,
        0.+0.j,  0.+0.236...j,  0.+0.414...j,  0.+0.535...j, 0.+0.618...j])    
    """
    t = np.sqrt(1+np.abs(x)**2)
    return x/(a+t)
    
def plane_to_sphere(x):
    """Stereographic projection"""
    newdim = list(x.shape)
    newdim[-1] = 3
    result = np.empty(newdim)
    t = 1/(1+np.sum(x**2, axis=-1) )
    result[...,:2] = 2 * x[..., :2]/t
    result[...,2] = (np.sum(x**2, axis=-1)-1)/t
    return result

def sphere_to_plane(y):
    """Stereographic projection"""    
    return y[...,:2] /(1-y[...,2])
    
def sphere_to_cylinder(sphcoords, zfunc=np.arcsin, scale=180/np.pi):
    """Projects 3d coordinates on the sphere onto a 2d rectangle.
    
    Args:
        sphcoords: An array of shape (..., 3).
        zfunc: A function to transform the z-values on the sphere. By 
               default this is np.arcsin, which makes the projection a
               "rectangular" map projection. Use zfunc = lambda x: x
               for an equal-area projection, and np.arctanh for Meractor.
        scale: A scale function, applied to both coordinates of the result. 
               By default this is 180/np.pi, to 
               transform radians into degrees.
    
    Returns:
        The 2d rectangular coordinates, in an array of shape (..., 2). 
        By default, returns latitude and longitude, but if zfunc is 
        specified, the second coordinate will be whatever the function
        transforms it to be.
        
    >>> x = np.random.normal(size=(10,3))
    >>> y = x/np.linalg.norm(x,axis=-1)
    >>> z = y - cylinder_to_sphere(sphere_to_cylinder(y))
    >>> np.abs(z).max() < 1E-15
    True       
    """
    #specify shape of result
    newdim = list(sphcoords.shape)
    newdim[-1] = 2
    result = np.empty(newdim)
    #populate the array
    result[...,0] = np.arctan2(sphcoords[..., 1], sphcoords[..., 0])*scale
    result[...,1] = zfunc(sphcoords[..., 2])*scale
    return result    
    
def cylinder_to_sphere(cylcoords, zfunc=np.sin, scale=180/np.pi): 
    """Projects a 2d rectangle onto the sphere.
    
    Args:
        cylcoords: An array of shape (..., 2).
        zfunc: A function to transform the z-values on the sphere. By 
               default this is np.sin, for "rectangular" map projections. Use 
               zfunc = lambda x: x for an equal-area projection, and 
               np.tanh for Meractor.
        scale: A scale function. By default this is 180/np.pi, to handle 
                coordinates in degrees.
    
    Returns:
        3d Euclidean coordinates on the sphere, in an array of shape (..., 3). 
        
    >>> x = np.random.normal(size=(10,3))
    >>> y = x/np.linalg.norm(x,axis=-1)
    >>> z = y - cylinder_to_sphere(sphere_to_cylinder(y))
    >>> np.abs(z).max() < 1E-15
    True

     """
    #specify shape of result
    newdim = list(cylcoords.shape)
    newdim[-1] = 3
    result = np.empty(newdim)
    #populate the array
    result[...,2] = zfunc(cylcoords[..., 1]/scale)    
    radius = np.sqrt(1-result[...,2]**2)
    result[...,0] = np.cos(cylcoords[..., 0]/scale)*radius
    result[...,1] = np.sin(cylcoords[..., 0]/scale)*radius

    return result     
    
def roll(avg,max,size=1):
    """ Basically a scaled Beta distribution that's well-behaved when 
    avg or max <=0, or when max = infinity."""    
    if max == np.inf and avg > 0:
        #the limit of the final case as max -> inf
        return np.random.gamma(avg, size=size)
    elif avg <= 0:
        return np.zeros(size)
    elif max <= avg:
        return max*np.ones(size)
    else:
        return max*np.random.beta(avg, max-avg, size=size)        
        
def uniform_disk(size=1):
    """Random values uniformly distributed on the complex unit disk.
    
    >>> x = uniform_disk(size=42)
    >>> np.all(np.abs(x) <= 1)
    True
    """
    return (np.sqrt(np.random.rand(size)) *
            np.exp(2j*np.random.uniform(0,np.pi,size)))     
            
def bivar_dist(size=1, dist = np.random.standard_normal):
    """Random samples from a complex bivariate distribution, based on
    a regular real distribution.
    
    By default this uses the standard normal. Standard cauchy is easily
    used. Standard t requires the df paramater to be partialed out. If you 
    want non-zero covariance, roll your own function.
    
    >>> """
    x = dist(size=(size, 2))
    return x.view(complex)[...,0]
        
#vectorized functions
def _vget(key, m, default=None):
    """ Vectorized getter for dicts. Performs a dict lookup for every 
    element in an array. Same grammar as the usual "get" for dicts. 
    
    >>> x = np.arange(5)
    >>> y = {1: 'one', 2: 'two', 3: 'three', 4: 'four'}
    >>> vget(x, y, default='Another number')
    array(['Another number', 'one', 'two', 'three', 'four'] ...)"""
    return m.get(key, default)     
    
vget = np.vectorize(_vget, excluded=["m","default"])

if __name__ == "__main__":

    
    import matplotlib.pyplot as plt
    thing = [(0.3,1),(3,10),(30,100),
             (0.7,1),(7,10),(70,100)]
    for a, m in thing:
        vals = roll(a, m, size=10000)
        print(np.mean(vals), np.std(vals), np.std(vals)/np.mean(vals))
        n, bins, patches = plt.hist(vals, 50)
        plt.show()    
    