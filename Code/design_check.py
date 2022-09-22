# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:02:43 2022

@author: Ishanki
"""
import math

def check_design_decisions(m, m2, PV_OPTIONS):
    counter = 0
    for s in m.S:
        for i in m.DES_res[s].i:
            diff_PV =  m2.DES_res[s].panels_PV_reals[i].value - m.DES_res[s].panels_PV_reals[i].value
            nlp_val = m2.DES_res[s].panels_PV_reals[i].value
            if not nlp_val.is_integer():
                counter +=1
            # print(diff_PV)
            if diff_PV<0:
                new_val = int(math.floor(m2.DES_res[s].panels_PV_reals[i].value))
                print(new_val)
                m2.DES_res[s].Y[i,new_val+1].fix(1)
            elif diff_PV>0:
                new_val = int(math.ceil(m2.DES_res[s].panels_PV_reals[i].value))
                if new_val+1 < PV_OPTIONS[-1]:
                    m2.DES_res[s].Y[i,new_val+1].fix(1)
                else:
                    m2.DES_res[s].Y[i,PV_OPTIONS[-1]].fix(1)
            else:
                new_val = int(round(m2.DES_res[s].panels_PV_reals[i].value))
                m2.DES_res[s].Y[i,new_val+1].fix(1)
    
    print(f"Indicator: {counter}")
    return counter