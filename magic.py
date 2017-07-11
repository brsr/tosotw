# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:18:55 2016

@author: Bstone
"""
import numpy as np
import xmath

alignment_names = ["Aether", "Aether/Earth", "Earth/Aether",
                   "Earth", "Earth/Water", "Water/Earth",
                   "Water", "Water/Air", "Air/Water",
                   "Air", "Air/Fire", "Fire/Air",
                   "Fire", "Fire/Aether", "Aether/Fire"]

alignment_glyphs = ["Q", "Qe", "Eq",
                    "E", "Ew", "We",
                    "W", "Wa", "Aw",
                    "A", "Af", "Fa",
                    "F", "Fq", "Qf"]
               
strtocomplex = {"0":0,
                 "?":np.nan}
                 
strtocomplex.update({alignment_glyphs[i]: 
                      -np.exp(i*np.pi*2j/len(alignment_glyphs)) 
                      for i in range(len(alignment_glyphs))})                 

strtocomplex.update({alignment_names[i]: 
                      -np.exp(i*np.pi*2j/len(alignment_names)) 
                      for i in range(len(alignment_names))})                 

                 
def angletostr(a, glyph = True):
    if glyph:
        name = np.array(alignment_glyphs)
        zeroname="0"        
        infname = "?"
    else:
        zeroname="Neutral"        
        name = np.array(alignment_names )
        infname = "Unknown"
    values = np.linspace(0,2*np.pi,len(name),endpoint=False)
    
    angle = a%(2*np.pi)
    index = np.searchsorted(values, angle)-1 - len(name)//2
    result = np.where(np.isfinite(a), name[index], infname)
    return np.where(np.isnan(a), zeroname, result)

def complextostr(z, glyph = True):
    """
    >>> np.all([name == complextostr(strtocomplex[name])  
                for name in alignment_glyphs])
    True
    """    

    return angletostr(xmath.angle(z), glyph)
     
     
def defensebonus(dalign, aalign, maxangle=4/5*np.pi, offset=0.05):
    #           attacker
    #           0, 1, inf
    #defender 0 0, 0, 0 
    # (self)  1 0, x, 1
    #       inf 0, 1, 1
    
    dmag = np.abs(dalign)
    #amag = np.abs(aalign)
    diff = xmath.angle(aalign) - xmath.angle(dalign)         
    c = np.cos(diff - maxangle) 
    c = np.where(np.isfinite(c), c, -1)#worst case
    #c = np.where(np.isfinite(dmag), c, 1)

    #could use dmag**2 or just dmag here
    result = dmag**2/(dmag**2 + offset + 1 - c )
    result = np.where(np.isfinite(dmag), result, 1)
    result = np.where((dalign == 0) | (aalign == 0), 0, result)
    
    return result     

if __name__ == "__main__":
    for key, num in strtocomplex.items():
        print(key[:3], complextostr(num))
        

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import matplotlib as mpl
    
    
    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05
    
    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-2, 2 + dy, dy),
                    slice(-2, 2 + dx, dx)]
    
    z = defensebonus(x+1j*y,1)
    z2 = defensebonus(x+1j*y,1) - defensebonus(x-1j*y,1) 

    t = np.linspace(0,2*np.pi,100)
    cx = np.cos(t)
    cy = np.sin(t)

    t5 = np.linspace(0,2*np.pi,5,endpoint=False)
    cx5 = np.vstack((2*np.cos(t5), np.zeros(len(t5))))
    cy5 = np.vstack((2*np.sin(t5), np.zeros(len(t5))))

    
    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    z2 = z2[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())    
    levels2 = MaxNLocator(nbins=15).tick_values(z2.min(), z2.max())

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = mpl.cm.magma
    #norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    
    fig, (ax0) = plt.subplots(nrows=1)
    
    ax0.set_aspect("equal")  
    # contours are *point* based plots, so convert our bound into point
    # centers
    
    cf = ax0.contourf(x[:-1, :-1] + dx/2.,
                      y[:-1, :-1] + dy/2., z, levels=levels,
                      cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    ax0.plot(cx,cy)
    ax0.plot(cx5,cy5)
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)
    
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    
    plt.show()    
    
    fig, (ax0) = plt.subplots(nrows=1)
    
    ax0.set_aspect("equal")  
    # contours are *point* based plots, so convert our bound into point
    # centers
    
    cf = ax0.contourf(x[:-1, :-1] + dx/2.,
                      y[:-1, :-1] + dy/2., z2, levels=levels2,
                      cmap=cmap)
    fig.colorbar(cf, ax=ax0)
    ax0.plot(cx,cy)
    ax0.plot(cx5,cy5)
    ax0.set_xlim(-1, 1)
    ax0.set_ylim(-1, 1)    
    
    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    
    plt.show()     
                