# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:23:35 2016

@author: bstone
"""

import numpy as np
#import pandas as pd
import xmath
import magic

from functools import total_ordering

#-1 : not mapped
#0 : mapped but not visible
#(0,1): vaguely visible
#1 : details visible

@total_ordering
class Entity:
    priority = 0
    birth = np.nan
    alignment = 0
    vis_range = 0
    def __init__(self, world, loc):
        self.world = world
        self.loc = loc
        self.effects = []
        
    def __eq__(self, other):
        return self.priority == other.priority
                
    def __lt__(self, other):
        #note that this is BACKWARDS. 
        #Higher priority results in appearing earlier in the list.
        return self.priority < other.priority
        
    def __str__(self):
        return ("Entity: " + self.kind+ " at " + self.loc + 
                    " birth turn " +self.birth)

    @classmethod
    def place(cls, world, loc):
        world.entities.append(cls(world, loc))
        
    @staticmethod
    def cost(world, loc):
        return NotImplemented         

    @property
    def view(self):
        #do line-of-sight eventually
        return self.world.nbhood(self.loc, self.vis_range)
        
    @property
    def actions(self):
        return None

    def cycle(self):
        pass           
    
    def turn(self):
        pass
                    
class CaveMouth(Entity):                   
    glyph = chr(0x26AB)         
    def __init__(self, world, loc):
        super().__init__(world, loc)
        self.link = None               
        
    @classmethod    
    def place(cls, world, loc):
        otherloc = (loc + world.geometry.N) % len(world.cells)
        cave = [cls(world, loc), cls(world, otherloc)]
        for i in range(2):
            cave[i].link = cave[i-1]     
        world.entities.extend(cave)
        
    @staticmethod
    def cost(world, loc):
        walkable = world.terraintypes.walk
        locpair = (loc + world.geometry.N) % len(world.cells)
        walkable = (walkable[world.cells[loc]["terrain"]].values &
                    walkable[world.cells[locpair]["terrain"]].values)
        #restrict to one side just to make things simpler
        eligible = walkable & (world.cells[loc]["level"] == 0)
        return np.where(eligible, 1/world.cells[loc]["elevation"] + 
                        1/world.cells[locpair]["elevation"], np.inf)
        
class RuinedSettlement(Entity):
    glyph = u"\u1f3e0\u0337"    
    def __init__(self, world, loc, birth=np.nan):
        super().__init__(world, loc)
        self.birth = birth
        
    @staticmethod
    def cost(world, loc):
        pop = world.cells["food"][loc]
        return 100/pop
     
class DeepSettlement(Entity):  
    vis_range = 3
    glyph = u"\u1f3e0\u0337"
    
    @staticmethod
    def cost(world, loc):
        eligible = [world.terrainlookup["Void"],
                    world.terrainlookup["Rock"],
                    world.terrainlookup["Magma"]]
        landindex = (~np.in1d(world.cells["terrain"], eligible) & 
            world.cells["level"] == 1)

        landcells = np.nonzero(landindex)[0]
        x = world.heur(landcells[:,np.newaxis], loc)
        exclude = landcells[:,np.newaxis] == loc
        x[exclude] = np.inf
        mindist = np.min(x,axis=0)
        cost = 10/mindist + world.geometry.cells["xyz"][world.cells["cell"],2]**2 

        return np.where( world.cells["level"][loc] == 1, cost, np.inf)    
        
    @classmethod
    def place(cls, world, loc):
        outerwall = world.ring(loc, 3)
        inside = world.nbhood(loc, 2)
        allcells = np.union1d(outerwall,inside)
        #remove rivers/roads in area
        index = world.linksincells(allcells)
        world.links["river"][index] = np.nan
        world.links["road"][index] = np.nan
        
        world.cells["population"][allcells] = 0
        world.cells["alignment"][allcells] = 0        
        world.cells["terrain"][inside] = world.terrainlookup["Barren"]
        world.cells["terrain"][outerwall] = world.terrainlookup["Rock"]
        super().place(world, loc)
        
    def cycle(self):
        self.world.cells[self.loc]["alignment"] = 0         
                    
class Settlement(Entity):
    priority = np.inf
    def __init__(self, world, loc, radius=2):
        super().__init__(world, loc)
        self.radius = radius        
        self.occupations = {}
        self.occupation_target = {}
        
    @property
    def lands_index(self):
        return self.world.nbhood(self.loc,self.radius)

    @property
    def lands(self):
        return self.world.cells[self.lands_index]
        
    @property
    def vis_range(self):
        return self.radius + 1
        
    @property
    def population(self):
        return np.sum(self.lands["population"]) 
        
    @property        
    def glyph(self):
        return chr(0x1f3e0)      
        
    @property
    def alignment(self):
        return np.sum(self.lands["alignment"]*self.lands["population"])

    def turn(self):
        #first calculate how much work we get out of each profession
        pass
        #then spend the work points
            

    @staticmethod
    def cost(world, loc):
        adj = world.cells[loc]["adjacency"]
        nb = np.hstack((loc[:,np.newaxis], adj))
        foodlookup = world.terraintypes.food.to_dict()
        foods = xmath.vget(world.cells[nb]["terrain"], foodlookup)
        terr = world.terraintypes.loc[world.cells[loc]["terrain"]]
        food = np.sum(foods,  axis=-1)
        eligible = (terr.walk & (np.abs(terr.damage) < 0.5)).values
        return np.where(eligible, 10000/food, np.inf)        
        
class Abbey(Entity):
    radius = 1
    birth = np.nan
    vis_range = 2    
    glyph = chr(0x269D)
    def __init__(self, world, loc, alignment):
        super().__init__(world, loc)         
        self.alignment = alignment    
        
    def cycle(self):
        self.world.cells[self.loc]["alignment"] = self.alignment               
        
        
class AbbeyEarth(Abbey):
    def __init__(self, world, loc):
        super().__init__(world, loc, alignment = world.align_const["E"])
        
    @staticmethod        
    def cost(world, loc):
        rock = world.terrainlookup["Rock"]
        water = world.terrainlookup["Water"]
        terr = world.cells["terrain"][loc]                     
        walkable = world.terraintypes["walk"].loc[terr]   
        level = world.cells["level"]
        river = world.cellsinlinks(np.isfinite(world.links["river"]) ) 
        #print(river.shape)
        terradj = terr[world.cells["adjacency"]]
        good = np.sum(terradj == rock, axis=-1)
        bad = np.sum(terradj == water, axis=-1)
        index = (level == 1) & walkable.values & (good < 6)
        index[river] = False
        cost = np.where(index, bad-good, np.inf)
        return cost    
        
class AbbeyWater(Abbey):
    def __init__(self, world, loc):
        super().__init__(world, loc, alignment = world.align_const["W"])
        
    @staticmethod    
    def cost(world, loc):
        water = world.terrainlookup["Water"]
        terr = world.cells["terrain"][loc]
        walkable = world.terraintypes["walk"].loc[terr]
        terradj = terr[world.cells["adjacency"]]
        bj = np.sum(terradj == water, axis=-1)
        cost = np.where(walkable, 6/bj, np.inf)
        return cost
        
class AbbeyAir(Abbey):
    def __init__(self, world, loc):
        super().__init__(world, loc, alignment = world.align_const["A"])   
        
    @staticmethod    
    def cost(world, loc):
        eligible = [world.terrainlookup["Mountain"],
                             world.terrainlookup["Alpine"]]
        icells = world.cells[loc]                          
        latitude = np.abs(world.geometry.cells["proj"][world.cells["cell"],1])
        index = (np.in1d(icells["terrain"], eligible) 
                 & (icells["level"] == 0)
                 & (latitude < 70))
        cost = np.where(index, 1/world.cells[loc]["elevation"], np.inf)
        return cost    
        
class AbbeyFire(Abbey):
    def __init__(self, world, loc):
        super().__init__(world, loc, alignment = world.align_const["F"])
        
    @staticmethod    
    def cost(world, loc):
        eligible = world.terrainlookup["Volcano"]
        index = world.cells["terrain"][loc] ==  eligible
        cost = np.where(index,
                        world.geometry.cells["xyz"][world.cells["cell"],2]**2,
                        np.inf)
        return cost
        
class AbbeyAether(Abbey):
    def __init__(self, world, loc):
        super().__init__(world, loc, alignment = world.align_const["Q"]) 
        
    @staticmethod    
    def cost(world, loc):
        pop = world.cells["food"][loc]
        terr = world.terraintypes.loc[world.cells[loc]["terrain"]]  
        eligible = (terr.walk & (np.abs(terr.damage) < 0.5)).values
        return np.where(eligible, pop, np.inf)
        
#%%                    
        
pronouns = {
'he':   {'they':'he',   'them':'him',  'their':'his',   'theirs':'his', 
         'themself':'himself'},
'she':  {'they':'she',  'them':'her',  'their':'her',   'theirs':'hers', 
         'themself':'herself'},
'it':   {'they':'it',   'them':'it',   'their':'its',   'theirs':'its', 
         'themself':'itself'},
'sthey':{'they':'they', 'them':'them', 'their':'their', 'theirs':'theirs', 
         'themself':'themself'},
'they': {'they':'they', 'them':'them', 'their':'their', 'theirs':'theirs', 
         'themself':'themselves'}
}        
class Troop(Entity):
    
    def __init__(self, world, loc, birth, defaultspecs, specs, 
                 kind, owner=None):
        super().__init__(world, loc)                              
        self.birth = birth       
        self.owner = owner
        self.kind = kind
        self.final = {}
        for name in defaultspecs:
            item = specs.get(name, defaultspecs[name])
            if isinstance(item, str):
                self.__dict__[name] = item
            elif isinstance(item, list):
                self.final[name] = item[1]
                self.__dict__[name] = item[0]                
            else:
                self.final[name] = item
                self.__dict__[name] = item
                
        self.ap = float(self.act_points)
        self.hp = float(self.hit_points)
        
    @property
    def actions(self):
        if self.ap > 0:
            return ["Placeholder"]
        else:
            return None
        
    @property
    def priority(self):
        #the adjustment here is taking the integral of the curve of 
        #avg_att vs wpp, and clipping it with max_att
        avgatt = self.avg_att
        maxatt = self.max_att
        wpp = self.wpp
        if (wpp > 0) or (~np.isfinite(maxatt) and wpp > -1) :
            adjatt = avgatt/(1 + wpp)        
        elif ~np.isfinite(maxatt):
            #reserve infinity for cities
            adjatt = np.finfo(float).max
        elif (wpp < 0):
            adjatt = (avgatt + maxatt*wpp*(maxatt/avgatt)^(1/wpp))/(1 + wpp)
        else: #elif wpp == 0:
            adjatt = avgatt*(1 - np.log(avgatt/maxatt))

        return adjatt + self.hit_points + self.avg_def
        
    @property
    def wpf(self):
        #wound penalty factor                 
        return (self.hp/self.hit_points)**self.wpp 

    def turn(self):
        #1 point of xp per turn just for existing
        self.experience(1)
        #update stats after XP so we're not always below 100%
        self.hp = min(self.hp+self.hit_points*self.healing_rate, 
                      self.hit_points)
        self.ap = min(self.ap+self.act_points, self.act_points)

        
    def combat_turn(self):       
        self.combat_ap = min(self.combat_ap+self.act_points, 
                             self.act_points)        

    def experience(self, xp):
        factor = 2**(-xp/self.xpscale)
        for name in self.final:
            current = self.__dict__[name]
            final = self.final[name]
            self.__dict__[name] = final + factor *(current - final)

    def roll_attack(self, n=1):
        aa = self.avg_att*self.wpf
        return self.max_att*xmath.roll(aa, self.max_att, n) 
                             
    def roll_defense(self, att_alig=0, n=1):        
        ad = np.max(self.avg_def, magic.defensebonus(self.alignment, att_alig) )                                 
        return self.max_def*xmath.roll(ad, self.max_def, n)
                                 
    def __str__(self):
        return (self.kind + "-type troop, born {self.birth}".format(self=self))
        
        

    
def attack(attacker, defender, n=1):
    att = attacker.roll_attack(n=n)
    att_alig = attacker.alignment
    defense = defender.roll_defense(att_alig = att_alig, n=n)  
    damage = np.maximum(att-defense, 0)
    alig_glyph = magic.complextostr(att_alig)
    return damage, np.where(damage > 0, alig_glyph, "0")
       
troopspecs = {
"Default": {
    'alignment':    0,
    'act_points':   2, 
    'hit_points':   1, 
    'healing_rate': [0.1,0.11],    
    'vis_range':    1,
    'vision':       [1, 1.01], 
    'stealth':      0,    
    'att_range':    1,                 
    'avg_att':      0.3,  
    'max_att':      1,  
    'avg_def':      0.3,  
    'max_def':      1,
    'wpp':          1,
    'xpscale':      100,
    'glyph':        '�',
    'pronoun':      'sthey',
    'description':  "Default... if you're seeing this, " +
                    "somebody's troop specs are wrong"
    },
"Scout": {
    'act_points': [4, 7], 
    'hit_points': [1, 2], 
    'vis_range':  2,       
    'vision':     [1.1, 1.2],   
    'stealth':    [0, 0.1],            
    'avg_att':    [0.2, 0.5],  
    'max_att':    [1, 2],  
    'avg_def':    [0.2, 0.5],  
    'max_def':    [1, 2],
    'glyph': chr(0x1F3C3),
    'pronoun':    "sthey",
    'description': "A brave and fast scout."
    },
"Militia": {
    'act_points': [2, 4],
    'hit_points': [2, 4],
    'vision': [1, 1.1],
    'avg_att': [0.6, 1.2],
    'max_att': [2, 3],
    'avg_def': [0.6, 1.2],
    'max_def': [2, 3],
    'glyph': chr(0x1f6e1),
    'pronoun': "they",
    'description': "Footsoldiers drawn from the local population."
    },
"Regulars": {
    'act_points': [2, 4],
    'hit_points': [3, 4],
    'vision': [1, 1.1],
    'avg_att': [1, 1.2],
    'max_att': [3, 3],
    'avg_def': [1, 1.2],
    'max_def': [3, 3],
    'glyph': chr(0x2694),
    'pronoun': "they",
    'description': "Professional infantry of the regular army."
    },
"Polearms":{
    'att_range':  2, 
    'act_points': [2, 4],
    'hit_points': [3, 4],
    'vision': [1, 1.1],
    'avg_att': [1, 1.2],
    'max_att': [6, 6],
    'avg_def': [1, 1.2],
    'max_def': [3, 3],
    'glyph': u"\u2694\u0337",
    'pronoun': "they",
    'description': "Soldiers who carry polearms to increase their reach."    
    },
"Cavalry": {
    'act_points': [3, 6],
    'hit_points': [4, 6],
    'vision': [1, 1.1],
    'avg_att': [1, 1.2],
    'max_att': [3, 3],
    'avg_def': [1, 1.2],
    'max_def': [3, 3],
    'glyph': chr(0x1F3C7),
    'pronoun': "they",
    'description': "Soldiers who fight while mounted on horseback."
    },     
"Archers": {
    'att_range':  5,
    'act_points': [2, 4],
    'hit_points': [2, 4],
    'vision': [1, 1.1],
    'avg_att': [0.6, 1.2],
    'max_att': [2, 3],
    'avg_def': [0.6, 1.2],
    'max_def': [2, 3],
    'pronoun': chr(0x1F3F9),
    'description': "Soldiers who carry bows and arrows."
    },    
"Longbows": {
    'att_range':  10,
    'act_points': [2, 4],
    'hit_points': [2, 4],
    'vision': [1, 1.1],
    'avg_att': [0.6, 1.2],
    'max_att': [2, 3],
    'avg_def': [0.6, 1.2],
    'max_def': [2, 3],
    'glyph': u"\u1F3F9\u20D6",
    'pronoun': "they",
    'description': "Soldiers who carry longbows with longer reach than" +
                    "regular archers."
    },    
"Mounted Archers": {
    'att_range':  5,
    'act_points': [3, 6],
    'hit_points': [4, 6],
    'vision': [1, 1.1],
    'avg_att': [0.6, 1.2],
    'max_att': [2, 3],
    'avg_def': [0.6, 1.2],
    'max_def': [2, 3],
    'glyph': u"\u1F3C7\u20D6",
    'pronoun': "they",
    'description': "Archers on horseback."
    },     
"Supply Train":{
    'glyph': chr(0x1F683),
    'effect':"Supply Train",
    'description':"Porters and pack animals that carry supplies."
    },    
}

if __name__ == "__main__":
    import json
    source = json.dumps(troopspecs)
    specs = json.loads(source)
    
    default = specs["Default"]
    scout = Troop(None, None, None, default, specs["Scout"], "Scout")
    militia = Troop(None, None, None, default, specs["Militia"], "Militia")
    for i in range(4):
        scout.turn()
        militia.turn()

    m_att, _ = attack(scout, militia, n=10000)
    
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(m_att, 50)
    plt.show()    
    