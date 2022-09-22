# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:48:00 2022

@author: Ishanki
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd

# Generating binary cuts
def cut_pack_generator(milp, iteration, binary_cut_pack):
    seasonal_bin_cut = {}
    for s in milp.S:
        bin_cut = {}
        bin_cut['W'] = {k: value(v) for k, v in milp.DES_res[s].W.items()}
        bin_cut['Y'] = {k: value(v) for k, v in milp.DES_res[s].Y.items()}
        bin_cut['J'] = {k: value(v) for k, v in milp.DES_res[s].J.items()}
        bin_cut['U'] = {k: value(v) for k, v in milp.DES_res[s].U.items()}
        seasonal_bin_cut[s] = bin_cut
    binary_cut_pack[iteration] = seasonal_bin_cut
    
def cut_constraint_generator(m, iteration, binary_cut_pack, 
                             KEEP_BATTERY, KEEP_PV, KEEP_BOILER, KEEP_HPTK,
                             FAST_RESULTS,):
    if iteration>1:
        for s in m.S:
            m.DES_res[s].del_component(m.DES_res[s].cut_constraints)
            m.DES_res[s].del_component(m.DES_res[s].cut_constraints_index)
            # m.del_component( m.DES_res[s].cut_constraints_index_index_0)
    
    for s in m.S:
        m.DES_res[s].cut_constraints = ConstraintList()
    
    for iteration, dict1 in binary_cut_pack.items():
        # print(iteration)
        for season, dict2 in dict1.items():
            # print(season)
            expr = 0
            for var_name, dict3 in dict2.items():
                ## Batteries
                if KEEP_BATTERY == 1:
                    if var_name == 'W':
                        for i in m.DES_res[season].i:
                            for c in m.DES_res[season].c:
                                if dict3[i,c] == 1:
                                    expr += (1 - m.DES_res[season].W[i,c])
                                    # print(expr)
                                else:
                                    expr += m.DES_res[season].W[i,c]
                ## PVs considered continuous.
                # if KEEP_PV == 1:
                #     if var_name == 'Y':
                #         for i in m.DES_res[season].i:
                #             for v in m.DES_res[season].v:
                #                 if dict3[i,v] == 1:
                #                     expr += (1 - m.DES_res[season].Y[i,v])
                #                     # print(expr)
                #                 else:
                #                     expr += m.DES_res[season].Y[i,v]
                ## Tanks and HPs
                if KEEP_HPTK == 1:
                    if var_name == 'J':
                        for i in m.DES_res[season].i:
                            for p in m.DES_res[season].p:
                                for k in m.DES_res[season].k:
                                    if dict3[i,p,k] == 1:
                                        expr += (1 - m.DES_res[season].J[i,p,k])
                                        # print(expr)
                                    else:
                                        expr += m.DES_res[season].J[i,p,k]
                ## Boilers - don't generate or consume any electricity
                # if not FAST_RESULTS:
                #     if KEEP_BOILER == 1:
                #         if var_name == 'U':
                #             for i in m.DES_res[season].i:
                #                 for b in m.DES_res[season].b:
                #                     if dict3[i,b] == 1:
                #                         expr += (1 - m.DES_res[season].U[i,b])
                #                         # print(expr)
                #                     else:
                #                         expr += m.DES_res[season].U[i,b]
                                    
            # print(expr)
            m.DES_res[season].cut_constraints.add(expr>=1)