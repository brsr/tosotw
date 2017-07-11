# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:26:27 2015

@author: Bstone

python-colormath hcl color space
"""
import geo
import entity
import numpy as np
import pandas as pd
import heapq
import xmath
import magic
from scipy.spatial import ConvexHull
from warnings import filterwarnings
#from functools import lru_cache

_gpcelldtype =np.dtype([("cell",     int),
                      ("level",      int),
                      ("adjacency", int,      6),
                      ("incidence", int,      6),

                      ("elevation",  float),
                      ("terrain",    np.uint8),
                      ("population", float),
                      ("growth",     float),
                      ("alignment",  complex),
                      ("food", float),
                      ("resource", float)])
_gplinkdtype = np.dtype([("link",   int),
                      ("level",     int),
                      ("cells",     int,      2),
                      ("river",     float),
                      ("road",      float)])
                      
_linkeqdtype = np.dtype([("river",     float),
                         ("road",      float)])                        
#                          
#_entitydtype = np.dtype([("location", int),
#                         ("onlink", bool),
#                         ("type", object)])      



class World:
    align_const = magic.strtocomplex
    boundfactor = 1/7     
    diffusion_factor = 0.04   
    sigma = 0.01
    theta = 0.01
    growthrate = 0.01    
    
    def __init__(self, geometry, landtypes, levels=2):
        """Constructor for the game world.
        Args:
            geometry: Underlying geometry.
            landtypes: A table of terrain types.
            levels: Number of levels in the world (defaults to 2).
        """
        self.geometry = geometry
        self.levels = levels
        self.terraintypes = landtypes
        #land lookup
        self.terrainlookup = {name: i for i, name 
                                in landtypes.name.iteritems()}

        #initialize cells
        #should probably split the geometry/topology items from the
        #gamepiece items
        ncells = geometry.N
        nlinks = len(geometry.links)
        cells = np.recarray(ncells*levels, dtype=_gpcelldtype)
        grid = np.array(np.meshgrid(np.arange(ncells), 
                            np.arange(levels))).reshape(2,-1)
        cells["cell"]  = grid[0]
        cells["level"] = grid[1]
       
        cells["adjacency"] = np.tile(geometry.cells["adjacency"],(2,1))
        cells["adjacency"] += ncells*cells["level"][..., np.newaxis]
        cells["incidence"] = np.tile(geometry.cells["incidence"],(2,1))
        cells["incidence"] += nlinks*cells["level"][..., np.newaxis]
        
        zerolist = ["elevation", "terrain", "population", "food", 
                    "resource", "growth","alignment"]
        for e in zerolist:
            cells[e] = 0                            
        #initialize links
        links = np.recarray(nlinks*levels, dtype=_gplinkdtype)
        grid = np.array(np.meshgrid(np.arange(len(geometry.links)), 
                            np.arange(levels))).reshape(2,-1)
        links["link"]  = grid[0]
        links["level"] = grid[1] 
        links["cells"] = np.tile(geometry.links["cells"], (2,1))                 
        links["cells"] += ncells*links["level"][..., np.newaxis] 

        nanlist = ["road","river"]
        for e in nanlist:
            links[e] = np.inf
        self.cells = cells
        self.links = links
        
        self.pents = np.concatenate([np.array(geometry.pents, 
                                              dtype=int) + ncells*i 
                                        for i in range(levels)])
        self.entities = []
        
    def __str__(self):
        return ("World with {self.levels} levels, ".format(self=self)
                + "based on " + str(self.geometry))
                
                
                
    @property
    def entities_location(self):
        #should probably cache this or something
        entities = self.entities
        result = {}
        for e in entities:
            loc = e.loc
            try:
                #this doesn't result in a sorted list, but the first element
                #does have highest priority. if we ever care about total order,
                #might want to change this to bisect
                heapq.heappush(result[loc], e)
            except KeyError:
                result[loc] = [e]
        return result                
                
    def pathfind(self, pt1, pt2, tilecost):
        return self.astarpath(pt1, pt2, self.heur, tilecost)
        
    #@lru_cache
    def nbhood(self, origin, n=1):
        #this is kind of inefficient        
        origin = np.unique(np.array(origin))
        if n <= 0:
            return origin
        else:            
            nbpts = self.cells["adjacency"][origin]
            return np.union1d(self.nbhood(nbpts, n=n-1), origin)
        
    #@lru_cache
    def ring(self, origin, n=1, width=1):
        return np.setdiff1d(self.nbhood(origin, n), 
                            self.nbhood(origin, n-width), assume_unique=True)
            
        
    def astarpath(self, start, goal, heur, 
                  tilecost=lambda x, y: 1, relaxation = 0):
        #tiles to examine: 
        #0) f-score, 1) g-score, 2) tile index
        agoal = np.array(goal)
        openset = [(heur(start, agoal), 0, start)]
        closedset = list()#tiles already examined
        came_from = {}#path back
        while openset:
            #print(openset)
            current = heapq.heappop(openset)
            #print(current)
            closedset.append(current[2])
            if current[2] == goal:
                #we've hit the goal so return the path
                return _reconstruct(came_from, goal), current[1], closedset
            nbors = self.cells["adjacency"][current[2]]
            links = self.cells["incidence"][current[2]]
            costs = tilecost(nbors, links)            
            heurs = np.maximum(heur(nbors, agoal), 0)
            #print(costs, current[1])
            gs = (1-relaxation)*(current[1] + costs)
            fs = gs + heurs
            #print(gs, heurs, fs)
            for i in range(len(nbors)):
                neighbor = nbors[i]
                if (neighbor not in closedset and np.isfinite(fs[i]) and
                    neighbor not in came_from and neighbor is not None):
                        op = (fs[i], gs[i], neighbor)
                        heapq.heappush(openset, op)
                        came_from[neighbor] = current[2]
        #if above didn't return, then there is no path: return None
        return None, None, closedset

    def heur(self, a, b):
        cella = self.cells["cell"][a]
        cellb = self.cells["cell"][b]
        levela = self.cells["level"][a]
        levelb = self.cells["level"][b]
        return (self.geometry.celldist(cella, cellb)/(self.geometry.maxstep) 
                + np.abs(levela-levelb))

    def linksincells(self, cells, both=False):
        x = np.in1d(self.links["cells"][..., 0], cells)
        y = np.in1d(self.links["cells"][..., 1], cells)
        if both:
            return x & y
        else:
            return x | y
            
    def cellsinlinks(self, links):
        return np.unique(self.links["cells"][links])    
                 
    def pathlinks(self, path):
        if path is None:
            return
        for i in range(len(path)-1):
            x = path[i]
            y = path[i+1]
            if x > y:
                x,y = y,x
            cells = self.links["cells"]
            
            index = (cells[:,0] == x) & (cells[:,1] == y)
            yield index.nonzero()[0]        
            
    @property
    def alignment(self):
        # + entity alignments
        people = np.sum(self.cells["alignment"]*self.cells["population"])  
        ents = np.sum([ent.alignment for ent in self.entities])
        return people + ents           
            
    @property
    def cell_equilibria(self):
        return self.terraintypes[["food", 
                                  "resource"]].loc[self.cells["terrain"]]
                                  
    @property
    def link_equilibria(self):
        cells = self.links["cells"]
        terr = self.cells["terrain"][cells]
        elev = self.cells["elevation"][cells]
        elevdiff = np.abs(elev[...,0] - elev[...,1])
        road = (self.terraintypes.cost.loc[terr[...,0]].values
                + self.terraintypes.cost.loc[terr[...,1]].values)
        terrfactor = np.sum(terr == self.terrainlookup["Water"],axis=-1)
        river = np.maximum(30*elevdiff/(terrfactor + 1), 1) 
        river[terrfactor == 2] = 1
        result = np.array(river, dtype=_linkeqdtype)
        result["road"] = road
        return result
        
    def reset_vars(self):
        #cells
        cells = self.cells
        n = len(cells)
        ce = self.cell_equilibria
        walkable = self.terraintypes["walk"].loc[cells["terrain"]].values        
        
        cells["food"]       = ce["food"]
        cells["resource"]   = ce["resource"]
        cells["growth"]     = self.growthrate
        cells["population"] = (np.random.beta(1,9,n)*cells["food"])        
        cells["alignment"]  = 0#xmath.uniform_disk(n)             
        #damage = self.terraintypes["damage"].loc[cells["terrain"]].values        
        #absd = np.abs(damage)
        #cells["alignment"][absd > 0] = damage/absd
                
        cells["population"][~walkable] = 0
        #links
        links = self.links
        le = self.link_equilibria

        lks = self.linksincells(walkable.nonzero(), both=True)
        links["road"][lks] = le["road"][lks]
        
        index = cells["terrain"] == self.terrainlookup["Water"]
        lks = self.linksincells(index.nonzero(), both=True)
        links["river"][lks] =1       
    
    def grow_vars(self):
        cells = self.cells        
        links = self.links
        pop = cells["population"] 
        growth = cells["growth"]

        ce = self.cell_equilibria
        le = self.link_equilibria
        #equilibriation phase
        eql = [(cells["growth"],   self.theta,  self.sigma, 
                self.growthrate, self.growthrate),
               (cells["food"],     self.theta,  self.sigma, 
                ce["food"],       0),
               (cells["resource"], self.theta,  self.sigma, 
                ce["resource"],   0),
               (pop,               growth, self.sigma, 
                cells["food"],    0),
               (links["road"],     self.theta,  self.sigma, 
                le["road"],  np.inf),
               (links["river"],    self.theta,  self.sigma, 
                le["river"], np.inf)]
        for var, rate, sigma, eq, impute in eql:
            #clip the rate so we don't wind up with negatives
            x = var/eq
            cap = np.where(x > 1, 1 / np.log(x), rate)
            cap = cap.clip(0, np.e)
            rate = np.clip(rate,0,cap)
            #actual equilibriation                               
            var -= rate * eq * xmath.xlogx(x)
            var[np.isnan(var)] = impute
            #random factor
            var *= np.random.lognormal(sigma=sigma, size=len(var))
            
    def drift_vars(self):
        alig = self.cells["alignment"]
        #shrink the alignments a bit to prevent abs(alig) == 1 exactly
        shrink = 1 - np.finfo(alig.dtype).epsneg
        angle = np.angle(alig)
        x = np.where(np.abs(alig)>=1, 
                     shrink*np.exp(1j*angle), alig)
        drift = self.sigma*xmath.bivar_dist(size=len(x))
        step = xmath.plane_to_disk(drift + xmath.disk_to_plane(x))
        self.cells["alignment"] = np.where(np.isfinite(step), step, x)
            
    def diffuse_vars(self):
 
        cells = self.cells 
        links = self.links        
        incidence = cells["incidence"]
        alig = cells["alignment"]
        pop = cells["population"]
        cap = cells["food"]
        diffusion = self.diffusion_factor*(1/links["road"] + 
                                                1/links["river"])
                                                                                              
        #diffusion phase   
        aligpop = alig * pop        
        left = links["cells"][..., 0]
        right = links["cells"][..., 1]
        diff = -(diffusion
                 *(pop[left]*cap[right] - pop[right]*cap[left])
                 /np.sqrt(cap[left]*cap[right]))                 
    #             *np.sqrt(cap[left]*cap[right])
    #             *(dens[left] - dens[right])) 
    #which of these is more numerically stable? we'll figure out later
    #probably doesn't matter unless cap is << 1
        diff[~np.isfinite(diff)] = 0
        come_from = np.where(diff>0,right,left)         
        #cap to +/- boundfactor*pop of whichever side is getting taken from                 
        pop_bound = self.boundfactor*pop[come_from]
        diff = np.clip(diff,-pop_bound,pop_bound)
        
        directional = np.where(cells["adjacency"] <= 
                               np.arange(len(cells))[:,np.newaxis], -1, 1)
        diff_structured = diff[incidence] * directional
        
        diff_alig = diff * alig[come_from]    
        diff_alig_structured = diff_alig[incidence] * directional
        #block out the 6th one on the pentagons    
        diff_structured[self.pents,-1] = 0
        diff_alig_structured[self.pents,-1] = 0
        dsumalig = np.nansum(diff_alig_structured, axis=-1)    
    
        pop += np.nansum(diff_structured, axis=-1)    
        cells["alignment"] = (aligpop + dsumalig) / pop     
        alig[np.isnan(alig)] = 0
               
    def cycle(self):
        self.grow_vars()
        self.drift_vars()
        self.diffuse_vars()
        #might need a cleaning step in here to impute nans etc
        for ent in self.entities:
            ent.cycle()
#%%            

def _reconstruct(came_from, current):
    """Function for the A* algorithm to recreate paths."""
    try:
        return _reconstruct(came_from, came_from[current]) + [current]
    except KeyError:
        return [current]
        
def plates(world, nplates=50):
    cells = world.geometry.cells["xyz"]
    adjacency = world.geometry.cells["adjacency"]
    vsites = geo.normalize(np.random.randn(nplates,3))#voronoi sites
    plateh = np.random.choice([-1,0,1],size=nplates, p=[0.35,0.3,0.35])
    angv = np.random.normal(size=(nplates,3))#random angular velocities
    
    cdists = world.geometry.dist(cells[:,np.newaxis], vsites)
    indlow = np.argmin(cdists,axis=-1)#index of closest site
    
    angvs = angv[indlow]
    linv = np.cross(cells, angvs)
    
    nb_rel_linv = linv[adjacency] - linv[:,np.newaxis]
    displacement = cells[adjacency] - cells[:,np.newaxis]
    inward_v = np.sum(nb_rel_linv*displacement, axis=-1)
    pents = world.geometry.pents
    inward_v[pents,-1] = np.nan#block out the 6th one on the pentagons
    mean_inward_v = np.nanmean(inward_v, axis=-1)
    
    #using tanh to bring the high ends down towards +/-1
    ridges = np.tanh(mean_inward_v)

    world.cells["elevation"] = np.tile(plateh[indlow],2)
    world.cells["elevation"] = np.hstack([ridges,ridges])
    
def underpoles(world, scale=50, exponent=16):# less land at poles
    index = ((world.cells["level"] == 1 ) & 
                (world.geometry.cells["pole"][world.cells["cell"]]))
    world.cells["elevation"][index] += scale                
    #z = world.geometry.cells["xyz"][...,-1]
    #world.cells["elevation"] += np.hstack([np.zeros(world.geometry.N),
    #                                        scale*(z**exponent)])
                                           
def craters(world, meanncraters=5, meansize=4, offset=0.5 ):
    #random craters
    ncraters = np.random.poisson(meanncraters)
    cratercenters = np.random.randint(0, len(world.cells)-1, ncraters)
    cratersizes = np.random.geometric(1/meansize, ncraters)
    
    for i in range(ncraters):
        crater = world.ring(cratercenters[i], 
                            cratersizes[i], cratersizes[i]-1)
        world.cells["elevation"][crater] -= offset
        
def noise(world, scale=1/4):#add noise
    world.cells["elevation"] += np.random.normal(scale=scale, 
                                                 size=len(world.cells))
    
def smooth(world, weight=0.1):
    adjacency = world.cells["adjacency"]
    pents = world.pents
    enb = world.cells["elevation"][adjacency]
    enb[pents,-1] = np.nan#block out the 6th one on the pentagons
    #maxelev = np.max(world.cells["elevation"])
    #weight = np.clip(world.cells["elevation"]/maxelev,minweight,1)
    world.cells["elevation"] = (np.nanmean(enb, axis=-1)*weight + 
                                world.cells["elevation"]*(1-weight))        
                
def quantize_elevation(world,n=1024):
    #quantize elevation
    cells = world.cells
    t = np.linspace(0,1,n,endpoint=False)
    pandafy = pd.DataFrame(cells[["level","elevation"]])
    pctiles = pandafy.groupby("level").elevation.quantile(q=t)
    elevl = []
    for i in range(world.levels):
        levelcells = cells[cells["level"] == i]        
        elev = levelcells["elevation"]
        bins = (np.digitize(elev,pctiles[i])-0.5)/n
        elevl.append(bins)
    world.cells["elevation"] = np.hstack(elevl)
     
def rivers(world, start=.8, n=108):   
    cells = world.cells
    start_potential = cells["elevation"] >= start
    allindex = np.arange(len(cells))
    starts = np.random.choice(allindex[start_potential], 
                              size=n, replace=False)
    riverheads = list(starts)
    while riverheads:
        head = riverheads.pop()
        head_elev = cells["elevation"][head]
        nb      = cells["adjacency"][head]
        nb_edge = cells["incidence"][head]
        nb_elev = cells["elevation"][nb]
        am = np.argmin(nb_elev)
        next_cell = nb[am]
        next_elev = cells["elevation"][next_cell]
        edge = nb_edge[am]
        if head_elev < next_elev:
            world.cells["terrain"][head] = world.terrainlookup["Water"]
        elif (~np.isfinite(world.links["river"][edge])):
            world.links["river"][edge] = 2
            riverheads.append(next_cell)   

def parsealign(s):
    dstr = s.split("@")
    try:
        rotation = magic.strtocomplex[dstr[1]]  
        damage = float(dstr[0])*rotation            
    except (KeyError, IndexError):
        damage = 0
    return damage

def importterrain(file='terrain.csv'):    
    return pd.read_csv(file, index_col="id", 
                       converters={"damage":parsealign})  
     
def importterraintrans(file='terrain_designation.csv'):
    return pd.read_csv(file).to_records()
     
   
     
def baseterrain(w, tertable):
    cells = w.cells    
    for item in tertable:
        index = ((cells["level"] == item["level"]) & 
                 (item["elev_low"] < cells["elevation"]) &
                 (cells["elevation"] <= item["elev_high"]) &
                 (cells["terrain"] == 0))
        cells["terrain"][index] = w.terrainlookup[item["type"]]    
                    
def volcanoes(w, eligible_names = ["Hills", "Mountain","Volcano","Alpine"]):  
    #volcanoes
    elevation = w.cells["elevation"]
    adj = w.cells["adjacency"]
    e_nb = w.cells["elevation"][adj]
    peakedness = np.array((e_nb < elevation[...,np.newaxis]),dtype=int) 
    #peakedness[pents,-1] = 1
    peaks = np.min(peakedness, axis=-1).astype(bool)
    eligible_terrain = [w.terrainlookup[name] for name in eligible_names]
    select = ((w.cells["level"] == 0) & peaks
             & np.in1d(w.cells["terrain"], eligible_terrain) )          
    volcells = np.random.choice(w.cells["cell"][select], size=17)
    lowselect = volcells + w.geometry.N
    w.cells["terrain"][volcells] = w.terrainlookup["Volcano"]  
    w.cells["terrain"][lowselect] = w.terrainlookup["Magma"]  
    magmaindex = w.cells["terrain"] == w.terrainlookup["Magma"]
    #don't replace magma with rock
    lowsurr = np.setdiff1d(w.ring(lowselect), magmaindex.nonzero()[0])
    w.cells["terrain"][lowsurr] = w.terrainlookup["Rock"]
    #remove rivers/roads in magma
    magmaindex = w.cells["terrain"] == w.terrainlookup["Magma"]
    index = w.linksincells(magmaindex.nonzero()[0])
    w.links["river"][index] = np.nan
    w.links["road"][index] = np.nan
    
def cellswithriver(w):    
    river = (w.links["river"] >= 0) & np.isfinite(w.links["river"])
    rivercells = np.unique(w.links["cells"][river])
    rivercellsbool = np.in1d(np.arange(len(w.cells)), rivercells)
    return rivercellsbool
    
def deserts(w):    
    #designate deserts
    rivercellsbool = cellswithriver(w)
    equatorial = (~rivercellsbool & (w.cells["level"]==0) & 
                  ((w.cells["terrain"] == w.terrainlookup["Plain"]) |
                  (w.cells["terrain"] == w.terrainlookup["Hills"]) )&
                  (np.abs(w.geometry.cells["proj"][..., 1][w.cells["cell"]]) 
                   < 20))
    w.cells["terrain"][equatorial] = w.terrainlookup["Desert"]
                    
def terraintransform(w, index, transform):      
    for name in transform:
        cur_index = index & (w.cells["terrain"] == w.terrainlookup[name])
        w.cells["terrain"][cur_index] = w.terrainlookup[transform[name]]
              
                    
def trees(w, transform = {"Plain":    "Forest",
                          "Hills":    "Forest",
                          "Marsh":    "Swamp",
                          "Barren":   "Mountain"}):                    
    #scatter trees
    terraintransform(w, cellswithriver(w), transform)
    
def icecaps(w):    
    #topside polar regions
    polarregion = ((w.cells["level"]==0) & 
                   (np.abs(w.geometry.cells["proj"][..., 1][w.cells["cell"]]) 
                   >= 70))
    terraintransform(w, polarregion, {"Plain":"Tundra",
                                      "Hills":"Tundra",
                                      "Marsh":"Tundra",
                                      "Desert":"Tundra",
                                      "Barren":"Tundra",
                                      "Forest":"Taiga",
                                      "Swamp":"Taiga",
                                      "Mountain":"Alpine"})
    terraintransform(w, (w.cells["level"] == 0) & 
                    (np.abs(w.geometry.cells["proj"][..., 1][w.cells["cell"]]) 
                    >= 80), {"Water":"Ice"})
    
#%% move these later    
def walk_tilecost(x,y):
    terrtype = w.cells["terrain"][x]
    result = np.where(w.terraintypes["walk"].loc[terrtype].values,
                      w.links["road"][y],
                      np.nan)
    return result
    
def swim_tilecost(x,y):
    terrtype = w.cells["terrain"][x]
    result = np.where(w.terraintypes["swim"].loc[terrtype].values,
                      w.links["river"][y],
                      np.nan)
    return result 
    
def fly_tilecost(x,y):
    #moves through flyable tiles at constant rate of 1/link    
    terrtype = w.cells["terrain"][x]
    result = np.where(w.terraintypes["fly"].loc[terrtype].values,
                      1,
                      np.nan)
    return result    
    
def burrow_tilecost(x,y):
    #moves through burrowable tiles at constant rate of 4/link
    terrtype = w.cells["terrain"][x]
    result = np.where(w.terraintypes["burrow"].loc[terrtype].values,
                      4,
                      np.nan)
    return result 

def ghost_tilecost(x,y):
    #can move through anything but void at 1/link
    terrtype = w.cells["terrain"][x]
    return np.where(terrtype,1,np.nan)                      

        
def roads(world):
    sites = list(world.entities_location.keys)
    sitexyz = world.geometry.cells["xyz"][world.cells["cell"][sites]]
    sitepairs = []
    for i in range(world.levels):
        include = (world.cells["level"][sites] == i)
        includedsites = sites[include]
        hull = ConvexHull(sitexyz[include])
        edges_set = {tuple(np.sort(pair)) 
                     for pair in hull.simplices[..., :-1]}
        edges_set.update(tuple(np.sort(pair)) 
                         for pair in hull.simplices[..., [0, 2]])
        edges_set.update(tuple(np.sort(pair)) 
                         for pair in hull.simplices[..., 1:])
        o_sites = includedsites[np.array([pair for pair in edges_set])]
        sitepairs.append(o_sites)
    sitepairs = np.vstack(sitepairs)
    for row in sitepairs:
        #print(row)
        path, y, z = w.pathfind(row[0],row[1], tilecost=walk_tilecost)
        if path is not None:
            plinks = np.array(list(world.pathlinks(path))).flatten()
            #print(len(path), len(plinks))
            #print(path, plinks)
            world.links["road"][plinks] = 1
            
def optimize_entities(world, elist, iterations=10):
    cells = np.arange(len(world.cells))
    level = world.cells["level"]
    entlist = [i[0] for i in elist]
    #nlist = [i[1] for i in elist]
    
    #create a starting case
    sitelist = []
    allsites = []
    costlist = []
    for ent, n in elist:
        #to start, pick sites with lowest cost that 
        #haven't been picked already
        c = ent.cost(world, cells)
        costlist.append(c)
        ccopy = c.copy()
        ccopy[allsites] = np.nan        
        
        newsites = np.argsort(ccopy)[:n]
        #print(newsites)
        sitelist.append(newsites)
        allsites.extend(newsites)        
    #unravel the array        
    unravel = []
    for ent, cost, sites in zip(entlist, costlist, sitelist):
        for site in sites:
            unravel.append([ent, cost, site])

    unravel = np.array(unravel)            
    sites = unravel[..., 2]
    types = unravel[..., 0]
    #jiggle it around until the cost stabilizes
    for i in range(iterations):  
        for i in range(len(unravel)):          
            ent, cost, site = unravel[i]
            otherindex = sites != site
            #do a thompson-problem type thing to space out the points
            othersites = sites[otherindex].astype(int)
            othercities = sites[otherindex & 
                                (types == entity.Settlement)].astype(int)
            #print(othercities)
            dist = world.heur(othersites,cells[:,np.newaxis])
            #includes an offset so stuff doesn't wind up right next to each other
            distcostpiece = 1/np.maximum(dist-1,0)**2
            distcost = np.sum(distcostpiece,axis=-1)
            total = cost + distcost 
            #term to encourage the same # to appear on each level
            #otherlevel = level[othersites]
            othercitylevel = level[othercities]
            #force at least half the cities to be up top
            if issubclass(ent, entity.Settlement):
                cities_top = (np.sum(othercitylevel) + level
                                > len(othercities)/2)
                total += 1E6 * cities_top
            newsite = np.argmin(total)
            unravel[i,2] = newsite

    return unravel[...,[0,2]]
            
if __name__ == "__main__":
    filterwarnings("ignore", "invalid value", category=RuntimeWarning)
    filterwarnings("ignore", "divide by zero", category=RuntimeWarning)
    
    np.random.seed(1)
    terrain = importterrain()
                                
    geometry = geo.GeodesicGeometry(16,9)
    w = World(geometry, terrain)
    #elevation
    plates(w, 29)
    underpoles(w)
    craters(w)
    noise(w)
    for i in range(50):
        smooth(w,0.05)
    quantize_elevation(w)
    #things that depend on elevation
    rivers(w, start=0.2, n=1000)
    baseterrain(w, importterraintrans())
    #things that depend on initial terrain
    volcanoes(w)
    deserts(w)
    trees(w)
    icecaps(w)

    #map variables
    w.reset_vars()

    #entities        
    elist = [(entity.DeepSettlement, 1),
             (entity.Settlement, 9),
             (entity.AbbeyEarth, 1),
             (entity.AbbeyWater, 1),
             (entity.AbbeyAir, 1),
             (entity.AbbeyFire, 1),
             (entity.AbbeyAether, 1),
             
             (entity.RuinedSettlement, 37),
             (entity.CaveMouth, 17)]     
    
    entsites = optimize_entities(w, elist)
    for ent, site in entsites:
        ent.place(w, site)
        
    for i in range(50):
        w.cycle()          
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.collections import LineCollection
    from matplotlib.colors import colorConverter
    vertices = w.geometry.cells["vertices"]
    flatpolys = w.geometry.vertices["proj"][vertices]
        
    wrapmask = np.max(flatpolys[...,0],
                      axis=-1) - np.min(flatpolys[...,0], axis=-1) > 200 
    lhs = (flatpolys[...,0] <0)
    flatpolys[...,0][wrapmask[...,np.newaxis] & 
                     lhs] = flatpolys[...,0][wrapmask[...,
                                                      np.newaxis] & lhs] + 360

    def showterr():        
        cproj = w.geometry.cells["proj"]
        
        lproj = np.dstack([
        cproj[w.geometry.links["cells"][...,0]],
        w.geometry.links["proj"],
        cproj[w.geometry.links["cells"][...,1]] ])
    
        #cave_proj = w.geometry.cells["proj"][caves]
        #pathcells = w.cells[["cell","level"]][path]
        #path_proj = w.geometry.cells["proj"][pathcells["cell"]]
        
        river_under_water = w.linksincells(np.nonzero(w.cells["terrain"] == 
                                            w.terrainlookup["Water"])[0],
                                            both = True)
                                            
        entitydict = w.entities_location
        entitykeys = np.array(list(entitydict.keys()))
        
        items = w.entities_location.items()
        sites = np.array([i[0] for i in items])
        glyphs = np.array(['$'+i[1][0].glyph+'$' for i in items])
        
        
        entitycolormap = {entity.AbbeyEarth: 'brown',
                          entity.AbbeyWater: 'blue',
                          entity.AbbeyAir: 'y',
                          entity.AbbeyFire: 'r',
                          entity.AbbeyAether: 'cyan',
                          entity.Settlement: 'w',
                          entity.DeepSettlement: '0.25',                          
                          entity.RuinedSettlement: 'grey',
                          entity.CaveMouth: 'k',
                          entity.Troop: 'magenta'}
        for i in range(w.levels):    
            i_entities = w.cells["level"][sites] == i
            
            i_sites = sites[i_entities]
            #i_glyphs = list(glyphs[i_entities])
            entities_proj = w.geometry.cells["proj"][w.cells["cell"][i_sites]]
            entities = xmath.vget(i_sites, entitydict)
            entities_c = [entitycolormap[type(e[0])] for e in entities]
            #print()
            #i_road = roadcells[w.cells["level"][roadcells] == i] 
            #road_proj = w.geometry.cells["proj"][w.cells["cell"][i_road]]
            river_links = w.links["link"][(np.isfinite(w.links["river"])) & 
                                            (w.links["level"] == i)
                                            &~river_under_water]
            river = lproj[river_links].transpose(0,2,1)
            badriver = np.abs(river[...,0,0]-river[...,2,0]) > 200
            lhs = (river[...,0] <0)
            index = lhs & badriver[...,np.newaxis]
            river[index,0] = river[index,0] + 360
    
            road_links = (w.links["link"][(w.links["road"] < 2) & 
                                            (w.links["level"] == i)] )
            road = lproj[road_links].transpose(0,2,1)
            badroad = np.abs(road[...,0,0]-road[...,2,0]) > 200
            lhs = (road[...,0] <0)
            index = lhs & badroad[...,np.newaxis]
            road[index,0] = road[index,0] + 360
            
            color = terrain["color"][w.cells["terrain"][w.cells["level"]==i]]                              
                                        
            colorrgb=[colorConverter.to_rgb(name) for name in color]
    
    
            index = (w.cells["level"] == i)
            fig = plt.figure()
            fig.set_size_inches(15,6)                                           
            ax = plt.axes()
            ax.set_aspect("equal")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            coll = PolyCollection(flatpolys, edgecolors='none',
                                  array = w.cells["elevation"][index],
                                  cmap=mpl.cm.jet)
            coll.set(array=None, facecolors=colorrgb)
            ax.add_collection(coll)
            #plt.colorbar(coll)
            
    
            line_segments = LineCollection(road,
                                            linestyles = 'solid', 
                                            color="0.75")
    #                                        array=w.links["road"]
    #                                       [(np.isfinite(w.links["road"])) & 
    #                                        (w.links["level"] == i)])
            #ax.add_collection(line_segments)
            
            line_segments = LineCollection(river,
                                           linestyles = 'solid', 
                                           color = "b")
    #                                       array=w.links["river"]
    #                                       [(np.isfinite(w.links["river"])) & 
    #                                        (w.links["level"] == i)])
            ax.add_collection(line_segments)
            
            ax.plot()
            ax.scatter(x=entities_proj[...,0],y=entities_proj[...,1], 
                       c=entities_c)        
            plt.show()
            
    def showpop(array = w.cells["population"]):        
        for i in range(w.levels):                                             
            index = (w.cells["level"] == i)
            fig = plt.figure()
            fig.set_size_inches(15,6)                                           
            ax = plt.axes()
            ax.set_aspect("equal")
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            coll = PolyCollection(flatpolys, edgecolors='none',
                                  array = array[index],
                                  cmap=mpl.cm.magma)
            #coll.set(array=None, facecolors=colorrgb)
            ax.add_collection(coll)
            fig.colorbar(coll)
            plt.show()        