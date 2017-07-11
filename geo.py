# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:54:45 2015

@author: Bstone
"""

import itertools
import numpy as np
import pandas as pd
import xmath
#from functools import lru_cache
from scipy.spatial import ConvexHull
from numpy.linalg import norm        
 
_gpcelldtype =np.dtype([("xyz",       float,    3),
                      ("proj",      float,    2),
                      ("face",      np.uint8, 1),
                      ("order",     np.uint8, 1),
                      ("pole",      bool,     1),
                      ("adjacency", int,      6),
                      ("incidence", int,      6),
                      ("vertices",  int,      6)])
_gplinkdtype = np.dtype([("xyz",      float,    3),
                      ("proj",      float,    2),
                      ("cells",     int,      2),
                      ("length",    float,    1)])
_gpvertexdtype = np.dtype([("xyz",    float,    3),
                       ("proj",     float,    2)])


        
def normalize(vec):
    """Normalizes vectors in n-space.

    >>> normalize((0,4)) #doctest: +NORMALIZE_WHITESPACE
    array([ 0.,  1.])    
    >>> normalize(testpt)#doctest: +NORMALIZE_WHITESPACE
    array([[ nan,   0.,   0.,   0.,   0.,   0.],
           [ nan,   1.,   1.,   1.,   1.,   1.]])
    """
    avec = np.array(vec)
    nm = norm(avec, axis=-1)
    return avec/nm[...,np.newaxis]

def rotation_matrix(theta, n=3):
    """Counterclockwise rotation about the Z axis in 3-space

    >>> np.round(rotation_matrix(np.pi/2),4)#doctest: +NORMALIZE_WHITESPACE
        array([[ 0., -1.,  0.],
               [ 1.,  0.,  0.],
               [ 0.,  0.,  1.]])
    """
    result = np.eye(n)
    result[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta),  np.cos(theta)]])
    return result
               
class GeodesicGeometry:
    """Contains information about the geometry of a geodesic grid and its
    dual Goldberg polyhedron.
    The methods used to calculate everything are documented in
    http://en.wikibooks.org/wiki/Geodesic_grids : in particular, the
    "naive slerp" method is used.
    
    Attributes:
        topology: A string describing the topology, i.e., "Sphere".
            This is a class attribute.
        order: The order of the geodesic sphere, which specifies the 
            underlying polyhedron. Will always be 5 (tetrahedron) for now.
        freq: A 2-tuple (n, m) describing the frequency of subdivision of
            the underlying polyhedron.
        T: A fundamental number of the grid: equal to n**2 + n*m + m**2
        N: Number of vertices in the geodesic grid, or number of cells in
            the corresponding Goldberg polyhedron.
        pents: Index of all pentagonal cells (dual vertices with order 5).            
        maxstep: The largest spherical distance between two adjacent 
            vertices (in radians).
        cells: An array containing information about each cell.
            Has these fields:
            "xyz": 3d-coordinates of the center of each cell. The center of 
                each cell is a vertex of the dual grid.
            "proj": 2d-projection of the center of each cell.
                Longitude and latitude, in degrees.
            "face": Which face of the underlying polyhedron the
                cell lies on
            "order": Order of the cell. Most are 6: those at the 
                vertices of the underlying polyhedron are 5.
            "pole": Whether the cell is at the north or south pole.
            "adjacency": A 6-element subarray of which cells are adjacent to 
                this cell.
            "incidence": A 6-element subarray of which links connect this 
                cell to the adjacenct cells.
            "vertices": A 6-element subarray of which vertices lie at the 
                corners of this cell.
                These three fields are closely related: the link numbered
                in "incidence" corresponds to the cell numbered in 
                "adjacency", and the vertices numbered in "vertices" all lie 
                to the side of their respective link. The elements are sorted
                so that they are in order as they cycle around the cell,
                in either clockwise or counterclockwise order.
                Note that in cells whose order is less than 6, the final 
                element is the same as the first. It may be necessary to
                exclude those elements from your calculation.
        links: An array containing information about the links between
            each cell. Has these fields:
            "xyz": 3d-coordinates of the midpoint of each link.
            "proj": 2d-projection of the center of each cell.
                Longitude and latitude, in degrees.                
            "cells": Which cells the link connects. The cell with lower 
                index is always first, so you will find (6, 7) here instead
                of (7,6).
            "length": Length of the link. Spherical distance, in radians.        
        vertices: An array containing information about the vertices of each 
            cell (NOT the vertices of the dual grid!). Fields:
            "xyz": 3d-coordinates of the vertex.
            "proj": 2d-projection of the vertex. 
                Longitude and latitude, in degrees. 
    """
    topology = "Sphere"
    def __init__(self, n=11, m=7, order=5, kfactor=0.5):
        """Constructor for the geodesic sphere geometry.
        
        Args:
            n, m: Frequency of the division.
            order: Order of the division. Only 5 is currently implemented.
                5 by default.
            kfactor: Parameter to naive slerp method.
        """
        if order != 5:
            raise NotImplementedError("Only grid of order 5 is" + 
                                      "currently implemented")
        elif n < 1 or m < 0:
            raise ValueError("Frequency out of range")
        self.order=order
        self.freq=(n, m)
        T= n**2 + n*m + m**2
        N = 10*T+2
        self.T= T
        self.N= N
        pents = [0, T, 2*T, 3*T, 4*T, 5*T, 6*T, 7*T, 8*T, 9*T, 10*T, 10*T+1]
        self.pents = pents
        #define a regular grid of 2d points
        grid_2d = np.array(np.meshgrid(np.arange(0, n+m+1),
                               np.arange(0, n+m+1))).reshape(2, (n+m+1)**2).T
        #outline the bounding triangle
        outside = ((grid_2d[..., 1]*n > (n+m)*grid_2d[..., 0]) | 
            (grid_2d[..., 1]*(n+m) < m*grid_2d[...,0]) | 
            (m*grid_2d[..., 1] + grid_2d[..., 0]*n >= T))
        overlap = (grid_2d[..., 1]*(n+m) == m*grid_2d[..., 0]) & ~outside
        def gridtobary(grid, n, m):
            """convert grid points to barycentric coordinates"""
            T = n**2 + n*m + m**2
            mat = np.array([[-m,  n+m],
                            [n+m,  -n]])/T
            alpha = np.empty((grid.shape[0], 3))
            alpha[:, :2] = grid.dot(mat)
            alpha[:, 2] = 1-alpha[:, 0]-alpha[:, 1]
            return alpha
        alpha1 = gridtobary(grid_2d[~outside], n, m )
        alpha2 = gridtobary(grid_2d[~outside & ~overlap], n, m )
        assert (alpha1.shape[0] + alpha2.shape[0] == T)

        #create empty array for the approximate pts 
        #and their respective faces
        pts = np.empty((N, 3))
        pts[:] = float('nan')
        pts[-1] = [0, 0,  1]
        pts[-2] = [0, 0, -1]

        face = np.empty(N, dtype=np.int)
        face[-1] = 0
        face[-2] = 19
        #base triangles, with correct orientation and whatnot
        icosa_h = np.sqrt(1/5) # = cos omega
        icosa_r = np.sqrt(4/5)
        icosa_x_pos = 1/2 + np.sqrt(5)/10
        icosa_x_neg = 1/2 - np.sqrt(5)/10      
        omega = np.arccos(icosa_h)  
        p1 = np.array([[0,          icosa_x_pos,           icosa_x_pos],
                       [0, np.sqrt(icosa_x_neg), -np.sqrt(icosa_x_neg)],
                       [1,              icosa_h,               icosa_h]])
        p2 = np.array([[          icosa_x_pos,          icosa_x_pos, icosa_r],
                       [-np.sqrt(icosa_x_neg), np.sqrt(icosa_x_neg), 0],
                       [              icosa_h,             icosa_h, -icosa_h]])
        p3 = np.array([[         icosa_x_pos,          icosa_x_neg, icosa_r ],
                       [np.sqrt(icosa_x_neg), np.sqrt(icosa_x_pos), 0],
                       [             icosa_h,             -icosa_h, -icosa_h]])
        p4 = np.array([[ icosa_r,          icosa_x_neg,  0],
                       [       0, np.sqrt(icosa_x_pos),  0],
                       [-icosa_h,             -icosa_h, -1]])
        
       
        def naiveslerp(alpha, p, kfactor = kfactor):
            """Calculates the 'naive slerp' method"""
            center = normalize(np.sum(p, axis=0))
            b = np.sin(omega*alpha)/np.sin(omega)
            px_stg = b.dot(p)
            k = 1 - norm(px_stg, axis=-1)
            px_stg += kfactor*k[..., np.newaxis]*center[np.newaxis]
            return normalize(px_stg)
            
        #start with 1/5th of the sphere, then rotate it to create the rest
        pts[:alpha1.shape[0]] = naiveslerp(alpha1, p1.T)
        pts[alpha1.shape[0]:T] = naiveslerp(alpha2, p2.T)
        pts[T:T+alpha1.shape[0]] = naiveslerp(alpha1, p3.T)
        pts[T+alpha1.shape[0]:2*T] = naiveslerp(alpha2, p4.T)
        face[:alpha1.shape[0]] = 0
        face[alpha1.shape[0]:T] = 1
        face[T:T+alpha1.shape[0]] = 2
        face[T+alpha1.shape[0]:2*T] = 3
        for i in range(1,5):
            rm = rotation_matrix(i*2*np.pi/5)
            pts[i*2*T:(i+1)*2*T] = pts[:2*T].dot(rm)
            face[i*2*T:(i+1)*2*T] = face[:2*T]+4*i

        #start building the cells data frame

        cells = np.recarray(N,dtype=_gpcelldtype)
        cells.xyz = pts
        cells.face = face#not sure this is actually needed
        cells.order = 6
        cells.pole = False
        cells.order[pents] = order
        cells.pole[[10*T, 10*T+1]] = True

        #cylindrical projection for mapping
        cells.proj = xmath.sphere_to_cylinder(cells.xyz)      

        #adjacency - which nodes meet which nodes
        #incidence - which nodes meet which edges
    
        #faces via convex hull
        #you can do this with a lot of twiddly algebra to figure out what 
        #points neighbor what, but this is actually readable and should work 
        #just as well unless the point distribution is really weird
        hull = ConvexHull(cells.xyz)

        #non-duplicated edges of each triangle, with low-index node first 
        #there's a funky indexing trick way to do this
        edges_set = {tuple(np.sort(pair)) 
                     for pair in hull.simplices[..., :-1]}
        edges_set.update(tuple(np.sort(pair)) 
                         for pair in hull.simplices[..., [0, 2]])
        edges_set.update(tuple(np.sort(pair)) 
                         for pair in hull.simplices[..., 1:])
        links = np.recarray(30*T,dtype=_gplinkdtype)    
        links.cells = [pair for pair in edges_set]                      
        links.sort(order="cells")

        low  = cells[links.cells[...,0]].xyz
        high = cells[links.cells[...,1]].xyz
        links.xyz = normalize(low + high)
        links.proj= xmath.sphere_to_cylinder(links.xyz)
        links.length = self.dist(low, high)
        self.maxstep = links.length.max()

        bearingdtype = np.dtype([("link",     int),
                                 ("start",     int),
                                 ("neighbor", int),
                                 ("bearing",  float),
                                 ("arg",      int)])
        #sorting the links by their bearing wrt the base cell
        bearingdata = np.recarray(60*T, dtype=bearingdtype)

        bearingdata.link = np.tile(np.arange(30*T),2)
        #need both direction of the link
        bearingdata.start = np.hstack([links.cells[...,0],
                                        links.cells[...,1]])
        bearingdata.neighbor = np.hstack([links.cells[...,1],
                                            links.cells[...,0]])                                        
        #calculate bearing 
        start = cells[bearingdata.start].xyz
        neighbor = cells[bearingdata.neighbor].xyz    
        bearingdata.bearing = self.bearing(start, neighbor)
        #bearing breaks down at the poles, 
        #but the indexes are already in cyclic order there
        index = [bearingdata.start >= 10*T]
        bearingdata.bearing[index] = bearingdata.neighbor[index]
        bearingdata.sort(order=["start","bearing"])       
        #need to resort to Pandas to get this pivot
        df = pd.DataFrame(bearingdata)                         
        df["rank"] = (df.groupby('start').bearing.rank()-1)
        step2 = df[['start','neighbor',
                       "rank"]].set_index(['start',"rank"]).unstack()
        cells.adjacency = step2.values
    
        cells.adjacency[..., 5][pents] = cells.adjacency[..., 0][pents]
        #same thing for incidence
        step2 = df[['start','link',
                       "rank"]].set_index(['start',"rank"]).unstack()
        cells.incidence = step2.values
    
        cells.incidence[..., 5][pents] = cells.incidence[..., 0][pents]
        
        #cell vertices
        vertices = np.recarray(20*T, dtype=_gpvertexdtype)
        vertices.xyz = normalize(np.sum(cells.xyz[hull.simplices],
                                    axis=1))
        flatpts = xmath.sphere_to_cylinder(vertices.xyz)
        vertices.proj= flatpts
        
        #more pandas crap. could use numpy.lib.recfunctions.joinby
        #but we've already imported pandas so w/e.
        #this is kinda inefficient but overall negligible.       
        #since we don't know what order the simplices are in, 
        #list all cyclic permutations and update the cell's 
        #data frame piece by piece
        permlist = [p for p in itertools.permutations([0,1,2])]

        simplices = pd.DataFrame(hull.simplices, columns=[0,1,2])
        simplices.reset_index(inplace=True)
        
        vx = pd.DataFrame(-np.ones((N,6),dtype=int))
        for i in range(6):
            leftfields = ["index","adjacency_"+str((i-1)%6),
                          "adjacency_"+str(i)]           
            adj = pd.DataFrame(cells.adjacency[..., [i-1, i]],
                               columns = leftfields[1:])
            adj.reset_index(inplace=True)
            for perm in permlist:
                step = pd.merge(adj, simplices,
                                left_on=leftfields, right_on=perm)      
                step = step[["index",
                             "index_y"]].set_index("index").sort_index()
                step = step.rename(columns={"index_y":(i-1)%6})
                vx.update(step)
                
        cells.vertices = vx.values        
        cells.vertices[...,5][pents] = cells.vertices[...,0][pents]
        self.cells = cells        
        self.links = links
        self.vertices = vertices        

        
    def __str__(self):
        return ("Geometry of the Goldberg polyhedron GP_" 
                + "{self.order}({self.freq[0]}, {self.freq[1]})".format(self =
                                                                        self)
                + " having {self.N} cells".format(self=self))
        
    #@lru_cache
    def nbhood(self, origin, n=1):
        """Returns a neighborhood around the origin cells, consisting of
            all cells that are within a certain number of links of
            origin.
        
        Args:
            origin: An array-like of cell indexes.
            n: Number of links, by default 1 (the origin itself and all cells 
                adjacent to it)
            
        Returns:    
            An array of cell indexes.        
        """        
        #this is kind of inefficient
        nbpts = self.cells.adjacency.loc[origin]
        if n <= 0:
            return origin
        else:
            return np.union1d(self.nbhood(nbpts, n=n-1), origin)
        
    #@lru_cache
    def ring(self, origin, n=1, width=1):
        """ Returns a ring around the origin cells. Effectively a set 
            difference between one neighborhood of size n and a smaller one
            of size n-width.
            
        Args:
            origin: An array-like of cell indexes.
            n: Number of links, by default 1.
            width: Width of the ring. By default 1.
            
        Returns:    
            An array of cell indexes. """
        return np.setdiff1d(self.nbhood(origin, n), 
                            self.nbhood(origin, n-width), assume_unique=True)
        
    def celldist(self, pt1, pt2):
        """ Returns the distance between cell centers.
        Args:
            pt1, pt2: Arrays of cell indexes.
            
        Returns: Array of distances.
        """
        coord1 = self.cells.xyz[pt1]
        coord2 = self.cells.xyz[pt2]                
        return self.dist(coord1, coord2)
        
    def cellbearing(self, origin, destination, pole=None):
        """ Returns the bearing (angle) between cells. By default,
            the bearing is calculated with respect to the north pole.
            
        Args:
            origin: Cell index of origin point 
            destination: Cell index of destination point
            pole: Cell index of pole, or None to default to the north pole.
        
        Returns: Array of bearings.       
        """
        coord1 = self.cells.xyz[origin]
        coord2 = self.cells.xyz[destination]   
        if pole is None:
            return self.bearing(coord1, coord2)
        else:
            coordN = self.cell.coord.loc[pole]        
            return self.bearing(coord1, coord2, coordN)
            
    @staticmethod                
    def dist(x, y, axis=-1):
        """Spherical distance, i.e. central angle, between vectors.
        Args:
            x, y: Coordinates of points on the sphere.
            axis: Which axis the vectors lie along. By default, -1.
        Returns: Array of spherical distances.    
        
        >>> t = np.linspace(0,np.pi,5)
        >>> c = np.cos(t)
        >>> s = np.sin(t)
        >>> z = np.zeros(t.shape)
        >>> x = np.vstack((c,s,z))
        >>> y = np.vstack((c,z,s))
        >>> GeodesicGeometry.dist(x,y,axis=0)/np.pi*180      
        array([  0.,  60.,  90.,  60.,   0.])# doctest: +NORMALIZE_WHITESPACE
        """
        #technically this is not the most numerically sound way to do this.
        #if issues arise, can change it.
        return np.arccos(np.clip(np.sum(x*y, axis=axis), -1, 1))
        
    @staticmethod
    def area(a,b,c):
        """Spherical area, i.e. solid angle. Note there are two areas defined
        by three points on a sphere: inside the triangle and outside it. This
        will always return the smaller of the two.
        
        Args:
            a, b, c: Coordinates of points on the sphere.
            
        Returns: Array of spherical areas.
        
        >>> x = np.array([1,0,0])
        >>> GeodesicGeometry.area(x,np.roll(x,1),np.roll(x,2))/np.pi
        0.5
        """
        top = np.abs(np.sum(a* np.cross(b, c), axis=-1))
        bottom = (1 + np.sum(a*b, axis=-1)
                    + np.sum(b*c, axis=-1)
                    + np.sum(c*a, axis=-1))
        return 2*(np.arctan(top/bottom)%np.pi) 
        
    @staticmethod                
    def bearing(origin, destination, pole = np.array([0,0,1])):
        """ Returns the bearing (angle) between points. By default,
            the bearing is calculated with respect to the north pole.
            Can also be considered as the angle adjacent to origin in the 
            triangle formed by origin, destination, and pole.
            
        Args:
            origin: Origin points
            destination: Destination points
            pole: Point bearing is calculated with respect to.
                By default, the north pole.
        
        Returns: Array of bearings.
        
        >>> x = np.array([1,0,0])        
        >>> GeodesicGeometry.bearing(x,np.roll(x,1))/np.pi
        0.5
        """        
        c_1 = np.cross(origin, destination)
        c_2 = np.cross(origin, pole)
        sin_theta = np.sum(origin*np.cross(destination, pole), axis=-1)
        cos_theta = np.sum(c_1*c_2, axis=-1)
        return np.arctan2(sin_theta,cos_theta)
        
_nbhoodmask = 1-np.eye(3,dtype=bool)

_nblist = np.array([[ 1, 0,-1],
                    [ 0, 1,-1],
                    [-1, 1, 0],
                    [-1, 0, 1],
                    [ 0,-1, 1],
                    [ 1,-1, 0]],dtype=int)
_kernel = 1-np.eye(3, dtype=np.int8)  
_kernel[1,1]=-128              

def flattenhexgrid(coords):
    """
    Flattens 3d hex grid coordinates into 2d coordinates.
    
    >>> x = np.arange(5)
    >>> coords = np.vstack((x,np.zeros(5),-x)).T
    >>> flattenhexgrid(coords)
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 0.,  2.],
           [ 0.,  3.],
           [ 0.,  4.]])    
    """
    result = np.dstack([coords[..., 1]*np.sqrt(3)/2,
                         coords[..., 0]+coords[..., 1]/2])[0]             
    return result
    
def nearesthex(pt):
    """
    Determines the 3d coordinates of the nearest hex to an array of points.
    
    >>> t = np.arange(5)-0.5
    >>> coords = np.vstack((x,np.zeros(5),-x)).T
    >>> nearesthex(coords)    
    array([[ 0,  0,  0],
           [ 1,  0, -1],
           [ 2,  0, -2],
           [ 3,  0, -3],
           [ 4,  0, -4]])    
    """
    rpt = np.round(pt)
    diffpt = np.abs(rpt-pt)
    argmaxdiff = np.argmax(diffpt,axis=-1)
    tot = np.sum(rpt,axis=-1)
    rpt[np.arange(rpt.shape[0]),argmaxdiff] -= tot
    return np.array(rpt, dtype=int)
#%%    
#@lru_cache
def _nbmask(n=1):
    return np.triu(np.tril(np.ones([2*n+1,]*2, dtype=bool),n),-n)[::-1]

#@lru_cache          
def _ringmask(n=1, j=1):
    outer = _nbmask(n).copy()
    inner = _nbmask(n-j)
    outer[j:-j,j:-j] -= inner
    return outer 
                   
def _masktoxyz(mask):
    """
    assumes mask is an indicator matrix on xy coords
    also that the center of the matrix is the center of the xyz
    """
    xy = np.array(mask.nonzero())
    cp = np.array(mask.shape)//2
    xy = xy - cp[..., np.newaxis]
    z = - xy[0] - xy[1]
    return np.vstack([xy,z])

#@lru_cache    
def _nbxyz(n=1):
    return _masktoxyz(_nbmask(n))
    
#@lru_cache    
def _ringxyz(n=1):
    return _masktoxyz(_ringmask(n))
    
_flatcelldtype =np.dtype([("xyz",       int,    3),
                      ("proj",      float,    2),
                      ("adjacency", int,      6),
                      ("incidence", int,      6),
                      ("vertices",  int,      6)])
_flatlinkdtype = np.dtype([("xyz",      float,    3),
                      ("proj",      float,    2),
                      ("cells",     int,      2)])
_flatvertexdtype = np.dtype([("xyz",    float,    3),
                       ("proj",     float,    2)])    
    
#%%                             
class FlatGeometry:
    """Contains information about the geometry of a flat triangular tiling and 
    its dual hexagonal tiling. This is all based on the Red Blob Game articles.
    
    Attributes:
        topology: A string describing the topology, i.e., "Disk".
            This is a class attribute.
        order: The order of the tiling: always 6 for hexes in the plane.
        freq: A 2-tuple (n, m) describing how many hexes along each skew-axis.
        N: Number of hexes in the grid.
        cells: An array containing information about each cell.
            Has these fields:
            "xyz": 3d-coordinates of the center of each cell. The center of 
                each cell is a vertex of the dual grid.
            "proj": 2d-projection of the center of each cell.
            "adjacency": A 6-element subarray of which cells are adjacent to 
                this cell.
            "incidence": A 6-element subarray of which links connect this 
                cell to the adjacenct cells.
            "vertices": A 6-element subarray of which vertices lie at the 
                corners of this cell.
                These three fields are closely related: the link numbered
                in "incidence" corresponds to the cell numbered in 
                "adjacency", and the vertices numbered in "vertices" all lie 
                to the side of their respective link. The elements are sorted
                so that they are in order as they cycle around the cell,
                in either clockwise or counterclockwise order.
                Note that cells along the edges may have a value of -1 for
                adjacent cells that do not exist.
        links: An array containing information about the links between
            each cell. Has these fields:
            "xyz": 3d-coordinates of the midpoint of each link.
            "proj": 2d-projection of the center of each cell.     
            "cells": Which cells the link connects. The cell with lower 
                index is always first, so you will find (6, 7) here instead
                of (7,6).        
        vertices: An array containing information about the vertices of each 
            cell (NOT the vertices of the dual grid!). Fields:
            "xyz": 3d-coordinates of the vertex.
            "proj": 2d-projection of the vertex. 
    """    
    
    topology = "Disk"
    order = 6
    nbmask = _nbhoodmask
    kernel = _kernel
    def __init__(self, n=25, m=25):
        """Constructor for the planar hex geometry.
        
        Args:
            n, m: Number of hexes along each skew-axis.
        """        
        
        if n < 1 or m < 1:
            raise ValueError("Frequency out of range")    
        self.freq = (n, m)
        N = n*m
        self.N = N
        
        cells = np.recarray(N, dtype=_flatcelldtype)
        cells.adjacency = -1
        cells.incidence = -1
        cells.vertices = -1
        
        #use the cube coordinate method
        grid = np.array(np.meshgrid(np.arange(0, n)-np.floor(n/2),
                                    np.arange(0, m)-np.floor(m/2),
                                    0)).reshape(3, (n*m)).T
        grid[...,2] = -grid[...,0]-grid[...,1]
        cells.xyz = grid
        cells.proj = flattenhexgrid(cells.xyz)     
        #the mask method for neighborhoods is quicker but we need 
        #the adjacency etc. so we can plot things
        nbhoods = cells.xyz[np.newaxis] + _nblist[:,np.newaxis]                          

        cols = ["x","y","z"]
        coordtable = pd.DataFrame(cells.xyz, columns=cols)
        coordtable.reset_index(inplace=True)
        for i in range(6):
            nbi = pd.DataFrame(nbhoods[i],columns=cols)
            nbi.reset_index(inplace=True)
            nbhoods_3 = pd.merge(nbi, coordtable, on=cols, how="left")
            result = nbhoods_3.index_y.values
            index = np.isfinite(result)
            cells.adjacency[index,i] = result[index]
            
        #links        
        index = np.arange(N)
        adjs = [cells.adjacency[...,i] for i in range(6)]
        linklist = np.vstack([np.tile(index,6), np.hstack(adjs)]).T 
         
        index = ((linklist[...,0] > linklist[...,1]) 
                    | (linklist[...,0] == -1) 
                    | (linklist[...,1] == -1))
        linklist = linklist[~index]
                    
        links = np.recarray(linklist.shape[0], _flatlinkdtype)            
        links.cells = linklist
        #link midpoints
        links.xyz = np.sum(cells.xyz[links.cells], 1)/2
        links.proj = flattenhexgrid(links.xyz)
        #cell - link correspondence
        
        clinks = pd.DataFrame(links.cells, columns=["l","h"])
        clinks.reset_index(inplace=True)
        for i in range(6):
            incidence = pd.DataFrame(-np.ones(N), columns=["incidence"])

            df = pd.DataFrame(cells.adjacency[..., i], columns=["adj"])
            df = df.reset_index()
            step = pd.merge(df, clinks, left_on=["index","adj"], 
                            right_on=["l","h"])
            step.set_index("index_x",inplace=True)
            incidence.update(step)
            #other direction
            step = pd.merge(df, clinks, left_on=["index","adj"], 
                            right_on=["h","l"])
            step.set_index("index_x",inplace=True)                            
            incidence.update(step)   
            cells.incidence[...,i] = np.squeeze(incidence.values)        
        
        #cell vertices
        #scale everything by 3 so we can use integer arithmetic
        cellvs = np.roll(_nblist,1,axis=0) + _nblist
        x = 3*cells.xyz[:,np.newaxis] + cellvs
        iterables = [np.arange(N),np.arange(6)]
        mindex =pd.MultiIndex.from_product(iterables,
                                           names=["cell","nb"])
        vertices2 = pd.DataFrame(x.reshape(-1,3), 
                                columns=["x","y","z"],
                                index = mindex)
                       
        vertices3 = vertices2.drop_duplicates(subset=["x","y","z"]).copy()

        vertices3.sort_values(["x","y","z"],inplace=True)      
        vertices3.index=np.arange(vertices3.shape[0]) 
        
        vertices = np.recarray(vertices3.shape[0], _flatvertexdtype)
        vertices.xyz = vertices3/3 
        vertices.proj = flattenhexgrid(vertices.xyz)        
        
        vertices3.reset_index(inplace=True)
        vertices2.reset_index(inplace=True)
        vx = pd.merge(vertices2, vertices3, on=["x","y","z"])        
        vx.set_index(["cell","nb"],inplace=True)       
        vx.sort_index(inplace=True)
        vx.index = mindex
        cells.vertices = vx["index"].unstack()
            
        self.cells = cells        
        self.links = links
        self.vertices = vertices
                                           
    def __str__(self):
        return ("Geometry of a diamond-shaped "
                + "{self.freq[0]} by {self.freq[1]} ".format(self=self)
                + "hex tiling having {self.N} cells".format(self=self))
                
    def ilookup(self, pts):
        """Looks up a cell index from its xyz coordinate.
        Args:
            pts: xyz coordinates of cells.
        Returns:
            An array of cell indexes.
        """
        index = (pts == self.cells.xyz)
        return index.nonzero()
        
    def nbhood(self, origin, n):
        """Returns a neighborhood around the origin cells, consisting of
            all cells that are within a certain number of links of
            origin.
        
        Args:
            origin: An array-like of cell indexes.
            n: Number of links, by default 1 (the origin itself and all cells 
                adjacent to it)
            
        Returns:    
            An array of cell indexes.   """        
        return self.ilookup(_nbxyz(n) + origin)

    def ring(self, origin, n=1, width=1):
        """ Returns a ring around the origin cells. Effectively a set 
            difference between one neighborhood of size n and a smaller one
            of size n-width.
            
        Args:
            origin: An array-like of cell indexes.
            n: Number of links, by default 1.
            width: Width of the ring. By default 1.
            
        Returns:    
            An array of cell indexes. """	        
        
        return self.ilookup(_ringxyz(n,width) + origin)
              
    def celldist(self, pt1, pt2, p = 2):
        """ Returns the distance between cell centers.
        Args:
            pt1, pt2: Arrays of cell indexes.
            p: Which norm to use. 2 = euclidean, 1 = taxicab.            
            
        Returns: Array of distances.
        """        
        coord1 = self.cells.xyz[pt1]
        coord2 = self.cells.xyz[pt2]
        return self.dist(coord1, coord2, p= p)
        
    def cellbearing(self, origin, destination, direction=None):
        """ Returns the bearing (angle) between cells. By default,
            the bearing is calculated with respect to the +y direction.
            
        Args:
            origin: Cell index of origin point 
            destination: Cell index of destination point
            direction: Direction vector, or None to default to the 
                +y direction.        
        Returns: Array of bearings."""       
        coord1 = self.cells.proj[origin]
        coord2 = self.cells.proj[destination]   
        if direction is None:
            return self.bearing(coord1, coord2)
        else:
            return self.bearing(coord1, coord2, direction)
              
    @staticmethod
    def dist(x, y, p = 2, axis=-1):
        """Distance between points.
        Args:
            x, y: Coordinates of points.
            p: Which norm to use. 2 = euclidean, 1 = taxicab.
            axis: Which axis the vectors lie along. By default, -1.
        Returns: Array of distances.    
        
        >>> t = np.linspace(0,1,5)[:,np.newaxis]
        >>> x = np.array([[0,0,0]])*t+np.array([[0,10,-10]])*(1-t)
        >>> y = np.array([[0,0,0]])*t+np.array([[10,0,-10]])*(1-t)        
        >>> FlatGeometry.dist(x, y)
        array([ 10. , 7.5, 5. , 2.5, 0. ])# doctest: +NORMALIZE_WHITESPACE
        """        
        if p == 2:
            return np.sqrt(np.sum((x-y)**2, axis=axis)/2)   
        elif p == 1:
            return np.max(np.abs(x-y), axis=axis)
        else:
            return (np.sum(np.abs(x-y)**p, axis=axis)/2)**(1/p)
         
    @staticmethod
    def area(a, b, c):
        """Area of triangle given by a, b, and c.
        
        Args:
            a, b, c: Coordinates of points.
            
        Returns: Array of areas.
        
        
        >>> a = np.array([0,0,0])
        >>> b = np.array([1,0,-1])
        >>> c = np.array([-1,1,0])
        
        >>> FlatGeometry.area(a,b,c)*np.sqrt(3)
        1.5...
        """        
        
        ab = a - b
        ac = a - c
        return norm(np.cross(ab, ac))/2
        
    @staticmethod                
    def bearing(origin, destination, direction = np.array([0, 1])):
        """ Returns the bearing (angle) between points. By default,
            the bearing is calculated with respect to the +y direction.
            
        Args:
            origin: Origin points
            destination: Destination points
            direction: A vector giving the direction the bearing is 
                calculated with respect to. By default, [0, 1].
        Returns: Array of bearings.
        >>> a = np.array([0,0,0])
        >>> b = np.array([1,0,-1])
        >>> c = np.array([-1,1,0])
                
        >>> FlatGeometry.bearing(a,b,c)/np.pi*180 
        120...
        """            
        
        pv = destination-origin
        d = np.sum(pv*direction, axis=-1)
        x = norm(np.cross(pv, direction))
        return np.arctan2(x, d)

    @staticmethod
    def line(pt1, pt2, fuzz=np.array([1E-6, 1E-6, -2E-6])):
        """
        Returns cells that are on a line between pt1 and pt2.
        >>> FlatGeometry.line(np.array([0,0,0]), np.array([9,-1,-8]))
        array([[ 0,  0,  0],
               [ 1,  0, -1],
               [ 2,  0, -2],
               [ 3,  0, -3],
               [ 4,  0, -4],
               [ 5, -1, -4],
               [ 6, -1, -5],
               [ 7, -1, -6],
               [ 8, -1, -7],
               [ 9, -1, -8]])       
        """
        Ncells = FlatGeometry.dist(pt1, pt2, p=1)
        t = np.linspace(0,1,Ncells+1)[...,np.newaxis]
        rawpts = (pt1+fuzz) * (1-t) + pt2 * t
        return nearesthex(rawpts)
         
if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    #geodesic
    gp = GeodesicGeometry()
    print(gp)    
    flatpolys = gp.vertices.proj[gp.cells.vertices]
    
    wrapmask = np.max(flatpolys[...,0],axis=-1)-np.min(flatpolys[...,0],
                                                       axis=-1) > 200 
    lhs = (flatpolys[...,0] <0)
    flatpolys[...,0][wrapmask[...,np.newaxis] & 
                     lhs] = flatpolys[...,0][wrapmask[...,
                                                      np.newaxis] & lhs] + 360
    acoord = gp.cells.xyz
    d = gp.dist(acoord[:,np.newaxis],acoord[gp.cells.adjacency])
    avgdist = np.max(d, axis=-1)/np.min(d, axis=-1)
    fig = plt.figure()
    #fig.set_size_inches(20,8)
    ax = plt.axes()
    ax.set_aspect("equal")    
    print(flatpolys.shape)
    coll = PolyCollection(flatpolys, edgecolors='none', array = avgdist, 
                          cmap=mpl.cm.jet)
    ax.add_collection(coll)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.colorbar(coll)
    plt.show()   

    fd = FlatGeometry()

    print(fd)
    flatpolys = fd.vertices.proj[fd.cells.vertices]
    
    flatpolys = np.roll(flatpolys, 1, axis=-1)
    lhs = (flatpolys[...,0] <0)
    acoord = fd.cells.xyz
    d = fd.dist(acoord[:,np.newaxis],acoord[fd.cells.adjacency])
    avgdist = np.max(d, axis=-1)/np.min(d, axis=-1)
    fig = plt.figure()
    #fig.set_size_inches(20,8)
    ax = plt.axes()
    ax.set_aspect("equal")    
    coll = PolyCollection(flatpolys, edgecolors='none', 
                          array = avgdist, cmap=mpl.cm.jet)
    ax.add_collection(coll)
    
    ax.set_xlim(-20, 20)
    ax.set_ylim(-15, 15)
    plt.colorbar(coll)
    plt.show() 
    #%matplotlib inline