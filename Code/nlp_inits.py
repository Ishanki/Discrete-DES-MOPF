# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 19:13:07 2022

@author: Ishanki
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd

def nlp_inits(m, m2, load_phase, S_BASE, SLACK):
    for s in m2.S:
        P_values = {}
        m2.DES_res[s].annual_inv_PV.value = m.DES_res[s].annual_inv_PV.value
        m2.DES_res[s].annual_inv_B.value = m.DES_res[s].annual_inv_B.value
        m2.DES_res[s].annual_cost_grid.value = m.DES_res[s].annual_cost_grid.value
        m2.DES_res[s].export_income.value = m.DES_res[s].export_income.value
        m2.DES_res[s].annual_oc_PV.value = m.DES_res[s].annual_oc_PV.value
        m2.DES_res[s].annual_oc_b.value = m.DES_res[s].annual_oc_b.value
        m2.DES_res[s].annual_inv_HP.value = m.DES_res[s].annual_inv_HP.value
        m2.DES_res[s].annual_inv_Tank.value = m.DES_res[s].annual_inv_Tank.value
        m2.DES_res[s].annual_inv_S.value = m.DES_res[s].annual_inv_S.value
        m2.DES_res[s].annual_oc_S.value = m.DES_res[s].annual_oc_S.value
        m2.DES_res[s].carbon_cost.value = m.DES_res[s].carbon_cost.value
        m2.DES_res[s].gen_income.value = m.DES_res[s].gen_income.value
        for i in m2.DES_res[s].i:
            m2.DES_res[s].max_H_b[i].value = m.DES_res[s].max_H_b[i].value
            for t in m2.DES_res[s].t:
                m2.DES_res[s].E_grid[i,t].value = m.DES_res[s].E_grid[i,t].value
                m2.DES_res[s].E_PV_used[i,t].value = m.DES_res[s].E_PV_used[i,t].value
                m2.DES_res[s].E_PV_sold[i,t].value = m.DES_res[s].E_PV_sold[i,t].value
                P_values[i,t] = m.DES_res[s].E_PV_sold[i,t].value \
                               - m.DES_res[s].E_grid[i,t].value \
                               - sum(m.DES_res[s].E_grid_charge[i,t,c].value for c in m.DES_res[s].c)
                for b in m2.DES_res[s].b:
                    m2.DES_res[s].H_b[i,t,b].value = m.DES_res[s].H_b[i,t,b].value
                for p in m2.DES_res[s].p:
                    m2.DES_res[s].E_hp[i,t,p].value = m.DES_res[s].E_hp[i,t,p].value
                    for k in m2.DES_res[s].k:
                        m2.DES_res[s].m_hp[i,t,p,k].value = m.DES_res[s].m_hp[i,t,p,k].value
                        m2.DES_res[s].H_hp[i,t,p,k].value = m.DES_res[s].H_hp[i,t,p,k].value
                for k in m2.DES_res[s].k:
                    m2.DES_res[s].T_tank[i,t,k].value = m.DES_res[s].T_tank[i,t,k].value
                    m2.DES_res[s].H_charge[i,t,k].value = m.DES_res[s].H_charge[i,t,k].value
                    m2.DES_res[s].H_discharge[i,t,k].value = m.DES_res[s].H_discharge[i,t,k].value
                    m2.DES_res[s].H_loss[i,t,k].value = m.DES_res[s].H_loss[i,t,k].value
                for c in m2.DES_res[s].c:
                    m2.DES_res[s].E_stored[i,t,c].value = m.DES_res[s].E_stored[i,t,c].value
                    m2.DES_res[s].E_grid_charge[i,t,c].value = m.DES_res[s].E_grid_charge[i,t,c].value
                    m2.DES_res[s].E_PV_charge[i,t,c].value = m.DES_res[s].E_PV_charge[i,t,c].value
                    m2.DES_res[s].E_charge[i,t,c].value = m.DES_res[s].E_charge[i,t,c].value
                    m2.DES_res[s].E_discharge[i,t,c].value = m.DES_res[s].E_discharge[i,t,c].value
        # print(P_values[s])
        P_init = {(n,p,t):0 for n in m2.OPF_res[s].n for p in m2.OPF_res[s].p \
                  for t in m2.OPF_res[s].t}
        for load, phase in load_phase.items():
            for (bus,time), val in P_values.items():
                if load == bus:
                    P_init[bus, phase.lower(), time] = val/S_BASE
        for n in m2.OPF_res[s].n:
            for p in m2.OPF_res[s].p:
                for t in m2.OPF_res[s].t:
                    if n==SLACK:
                        P_init[n,p,t] = sum(P_init[n,p,t] for n in m2.OPF_res[s].n \
                                            if n!=SLACK)*-1
        for n in m2.OPF_res[s].n:
            for p in m2.OPF_res[s].p:
                for t in m2.OPF_res[s].t:
                    m2.OPF_res[s].P[n,p,t].value=P_init[n,p,t]
        
        # print(P_init[34,'a',14])
        with open('init2_check.txt','w') as f:
            for k, v in P_init.items():
                f.write(f'{k}: {v}\n')
            f.write('\n\n')


