# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:07:47 2022

@author: Ishanki
"""

def results_prints(m):
    print(f"\n")
    annual_grid = sum(m.annual_cost_grid.value for m in m.DES_res[:])
    print("annual_grid_cost = " + str(annual_grid))
    carb_grid = sum(m.carbon_cost.value for m in m.DES_res[:])
    print("grid_carbon_cost = " + str(carb_grid))
    annual_PV_inv_cost = sum(m.annual_inv_PV.value for m in m.DES_res[:])
    print("annual_PV_inv_cost = " + str(annual_PV_inv_cost))
    annual_PV_op_cost = sum(m.annual_oc_PV.value for m in m.DES_res[:])
    print("annual_PV_op_cost = " + str(annual_PV_op_cost))
    annual_boiler_inv_cost = sum(m.annual_inv_B.value for m in m.DES_res[:])
    print("annual_boiler_inv_cost = " + str(annual_boiler_inv_cost))
    annual_boiler_op_cost = sum(m.annual_oc_b.value for m in m.DES_res[:])
    print("annual_boiler_op_cost = " + str(annual_boiler_op_cost))
    annual_batt_inv_cost = sum(m.annual_inv_S.value for m in m.DES_res[:])
    print("annual_battery_inv_cost = " + str(annual_batt_inv_cost))
    annual_batt_op_cost = sum(m.annual_oc_S.value for m in m.DES_res[:])
    print("annual_battery_op_cost = " + str(annual_batt_op_cost))
    annual_ashp_inv_cost = sum(m.annual_inv_HP.value for m in m.DES_res[:])
    print("annual_ashp_inv_cost = " + str(annual_ashp_inv_cost))
    annual_tank_inv_cost = sum(m.annual_inv_Tank.value for m in m.DES_res[:])
    print("annual_tank_inv_cost = " + str(annual_tank_inv_cost))
    annual_inc = sum(m.export_income.value for m in m.DES_res[:])
    print("export_income = " + str(annual_inc))
    annual_FIT = sum(m.gen_income.value for m in m.DES_res[:])
    print("annual_FIT = " + str(annual_FIT))