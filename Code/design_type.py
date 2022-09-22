# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 09:21:34 2022

@author: Ishanki
"""


def deactivate_integer_design(m):
    ## PVs
    for s in m.S:
        m.DES_res[s].PVR1.activate()
        m.DES_res[s].PVR2.activate()
        m.DES_res[s].PVR3.activate()
        m.DES_res[s].PVR4.activate()
        m.DES_res[s].PVR5.activate()
        m.DES_res[s].PVR6.activate()
        m.PVLink2.activate()
        # m.DES_res[s].PVN1.deactivate()
        m.DES_res[s].PVN2.deactivate()
        m.DES_res[s].PVN3.deactivate()
        m.DES_res[s].PVN4.deactivate()
        m.DES_res[s].PVN5.deactivate()
        m.DES_res[s].PVN6.deactivate()
        m.DES_res[s].PVN7.deactivate()
        m.PVLink1.deactivate()
    
def activate_integer_design(m):
    for s in m.S:
        ## PVs
        # m.DES_res[s].PVN1.activate()
        m.DES_res[s].PVN2.activate()
        m.DES_res[s].PVN3.activate()
        m.DES_res[s].PVN4.activate()
        m.DES_res[s].PVN5.activate()
        m.DES_res[s].PVN6.activate()
        m.DES_res[s].PVN7.activate()
        m.PVLink1.activate()
        m.DES_res[s].PVR1.deactivate()
        m.DES_res[s].PVR2.deactivate()
        m.DES_res[s].PVR3.deactivate()
        m.DES_res[s].PVR4.deactivate()
        m.DES_res[s].PVR5.deactivate()
        m.DES_res[s].PVR6.deactivate()
        m.PVLink2.deactivate()
        
    