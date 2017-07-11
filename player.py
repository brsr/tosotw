# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 16:25:46 2016

@author: Bstone
"""

class Player():
    def __init__(self, name):
        self.name = name
        self.view = -1
        self.entities = []
        
    def turn(self):
        for ent in self.entities:
            ent.turn()