# -*- coding: utf-8 -*-
"""
Combining the unbalanced power flow model (NLP) with the DES MILP formulation.

@author: Ishanki
    
"""

from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from time import perf_counter
from pandas import ExcelWriter
from MILP_discHPsTKsBsr import ResidentialSeasonalDES
from DES_UPF_PU import UnbalancedPowerFlow
from DES_OPF import DES_OPF
from binary_cut_functions_upd import cut_pack_generator, cut_constraint_generator
from nlp_inits import nlp_inits
from rprints import results_prints
from design_type import deactivate_integer_design, activate_integer_design
from design_check import check_design_decisions

# =============================================================================
#                              '''Inputs here'''
# =============================================================================
# house = ['A', 'B', 'C','D','E']
loads_file_name = "profiles.xlsx" #"summer.xls" #"Loads_houses.xls"
parameters_file_name = "Parameters_new.xlsx" 
irrad_file_name = "Parameters_new.xlsx"
# ASHP water outlet temperature in Celcius:
T_out_ashp = 55 
# Tank options
tanks = ['V','S', 'M', 'L']
# duration of time interval:
interval = 1  # 0.5 
# State the final time step e.g. 24 or 48:
FT = 24  #48
# Number of seasons to run
SEASONS = 4
# Days in each season, 1-winter, 2-spring, 3-summer, 4-autumn:
d = {1:90, 2:92, 3:92, 4:91} 
# Results filename, looks incomplete but will attach the string of season later

# Activate/deactivate technologies by entering either 1 or 0.
KEEP_BATTERY = 0
KEEP_PV = 0
KEEP_BOILER = 0
KEEP_HPTK = 1
PV_OPTIONS = list(range(0,21)) #Integer options for PVs - upto 20 panels

# Switch OPF functionality off using 0 (when running MILP), else use 1:
# N.B. default solver is CPLEX for MILP, and CONOPT for NLP
KEEP_OPF = 1
SOLVER = "conopt"
KEEP_COMPLEMENTARITY=1
BINARY_CUTS = 1
FAST_RESULTS = True
# This timeout is for the binary cuts algorithm
TIMEOUT = 10*60*60  #in seconds

# distribution network
size = 'medmod'
tfind = 'wtfDY'
NETWORK_FILE_NAME = f'Network_{size}_{tfind}.xlsx'
# resultsfile = f'upf_{size}_{tfind}_{FT}.xlsx'

SLACK = 0
SLACK_ANGLES = [0, -120, 120]  # In degrees
SUBSTATION = SLACK #secondary slack where power values are reported
PHASE_SHIFT = -30
PRIMARY_BUSES = [0,999]     #Leave empty if voltages across trafo are the same

if KEEP_OPF == 1:
    if KEEP_COMPLEMENTARITY ==1:
        mstr = "NLP_comp"
        if BINARY_CUTS == 1:
            if FAST_RESULTS:
                mstr = mstr + "_fast"
            else:
                mstr = mstr + "_slow"
    else:
        mstr = "NLP"
else:
    mstr = "MILP"
results_file_name = f"{mstr}_{size}_{tfind}_B{KEEP_BATTERY}_PV{KEEP_PV}_BR{KEEP_BOILER}_HP{KEEP_HPTK}_results" 
results_file_suffix = '.xlsx'
bounds_file_name = f"Bounds_{mstr}_{size}_{tfind}_B{KEEP_BATTERY}_PV{KEEP_PV}_BR{KEEP_BOILER}_HP{KEEP_HPTK}" + results_file_suffix
# =============================================================================
#                           '''Data Processing'''
# =============================================================================
# df_loadsinfo = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads_info')
df_source = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Source')
df_buses = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Buses')
df_loads = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Loads')
df_lines = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Lines')
df_linecodes = pd.read_excel(NETWORK_FILE_NAME, sheet_name='LineCodes')
df_transformer = pd.read_excel(NETWORK_FILE_NAME, sheet_name='Transformer')
S_BASE = df_transformer.MVA[0]*1000
# house_bus_connections = dict(zip(df_loads.Bus, self.df_loads.phases))
houses = list(df_loads.Bus)
load_phase = dict(zip(df_loads.Bus, df_loads.phases))

# These dataframes are fed directly into the ResidentialSeasonalDES class
df_scalars = pd.read_excel(parameters_file_name, sheet_name = "Res_Scalars")
#print(df_scalars)
df_roof = pd.read_excel(parameters_file_name, sheet_name = "Roof_areas_res")
df_batteries = pd.read_excel(parameters_file_name, sheet_name = "batteries")
df_ashps = pd.read_excel(parameters_file_name, sheet_name='ASHPs')
df_tanks = pd.read_excel(parameters_file_name, sheet_name='Tanks')
# Dfs which have seasonal specific params
dfi = pd.read_excel(irrad_file_name, sheet_name = "Irradiance")
dfi.set_index("Irrad", inplace = True)
df_Ta = pd.read_excel(irrad_file_name, sheet_name="Air_temp")
df_Ta.set_index("Temp", inplace=True)
# df_options = pd.read_excel(parameters_file_name, sheet_name="HP_Tank_options",header=0)
# df_options=df_options.set_index('Label')
df_tank_costs = pd.read_excel(parameters_file_name, sheet_name="Tank_cost_HP",header=0)
df_tank_costs=df_tank_costs.set_index('Label')
df_boilers = pd.read_excel(parameters_file_name, sheet_name="Boilers",header=0)

# Battery options given:
battery = df_batteries.type.to_list()
ashps = df_ashps.Label.to_list()
tanks = df_tanks.Label.to_list()
boilers = df_boilers.Label.to_list()

# this is here to easily loop the data imported from excel
house_num = list(range(1,(len(houses))+1))

def tank_hp_options(df_tank_costs, ashps, tanks):
    index_tuples = [(p,k) for p in ashps for k in tanks]
    multi_ix = pd.MultiIndex.from_tuples(index_tuples)
    # df1 = pd.DataFrame(df_options[tanks].values.ravel(), index=multi_ix, columns=["data"])
    df2 = pd.DataFrame(df_tank_costs[tanks].values.ravel(), index=multi_ix, columns=["data"])
    # Indicator = df1.to_dict()["data"]
    Indicator = {}
    original_costs = df2.to_dict()["data"]
    Tank_costs = {}
    for k, v in original_costs.items():
        if np.isnan(v):
            Indicator[k] = 0
            Tank_costs[k] = 0
        else:
            Indicator[k] = 1
            Tank_costs[k] = v
    return Indicator, Tank_costs

[Indicator, Tank_costs] = tank_hp_options(df_tank_costs,ashps,tanks)

start = perf_counter()

# =============================================================================
# '''The residential block'''
# =============================================================================
def MILP(m):
    # Extracting and passing only the data relevant to each season
    def seasonal_data(m,s):
        m.df = {} #represents the dataframe for electricity loads
        m.dfh = {} #represents the dataframe for heating loads
        # Looping through the loads w.r.t each season s and house from excel
        for n, h in zip(house_num, houses):
            ## Elec
            sheet_n1 = (f"Elec_{n}")
            m.df[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n1)
            new_columns = m.df[n].columns.values
            new_columns[0] = 'Season'
            m.df[n].columns = new_columns
            m.df[n].set_index('Season', inplace=True)
            m.df[n] = m.df[n].iloc[s-1]
            ## Heat
            sheet_n2 = (f"Heat_{n}")
            m.dfh[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n2)
            new_columns = m.dfh[n].columns.values
            new_columns[0] = 'Season'
            m.dfh[n].columns = new_columns
            m.dfh[n].set_index('Season', inplace=True)
            m.dfh[n] = m.dfh[n].iloc[s-1]
            string1 = f"this is electricity for season {s} for house {h} "
            string2 = f"this is heat for season {s} for house {h} "
            # print(string1)
            # print(m.df[n])
            #print(b.df[n].get(2))
            
        # Assigning loaded dataframes into dictionaries, now w.r.t house h and time t
        # print("data is now loading into loops")
        m.elec_house = {}
        m.heat_house = {}
        for n, h in zip(house_num, houses):
            for t in range(1,FT+1):
                m.elec_house[h, t] = round(float(m.df[n][t]/interval),5) 
                m.heat_house[h, t] = round(float(m.dfh[n][t]/interval),5)
                    
        #print(m.elec_house['h1', 34])
        #print(m.heat_house['h2',34]) 
        
        # Loading other time-dependent parameters
        m.dfi = dfi.iloc[s-1]
        m.df_Ta = df_Ta.iloc[s-1]
        m.irrad = {}
        m.T_amb = {}
        for t in range(1,FT+1):
            m.irrad[t] = float(m.dfi[t])
            m.T_amb[t] = float(m.df_Ta[t])
        # print(m.irrad)
        # print(m.T_amb)
    
        m.days = d[s]
        
        # The object m.full_model is created from the ResidentialSeasonalDES class
        m.full_model = ResidentialSeasonalDES(house=houses, 
                                          df=m.df, 
                                          days=m.days,
                                          interval=interval, 
                                          ft=FT, 
                                          irrad=m.irrad,
                                          df_scalars=df_scalars,
                                          df_roof=df_roof,
                                          elec_house=m.elec_house,
                                          heat_house=m.heat_house,
                                          df_batteries=df_batteries,
                                          # df_volume=df_volume, 
                                          battery=battery,
                                          df_loads=df_loads,
                                          SEASONS=SEASONS,
                                          KEEP_BATTERY=KEEP_BATTERY,
                                          KEEP_PV=KEEP_PV,
                                          PV_OPTIONS=PV_OPTIONS,
                                          ashps=ashps,
                                          df_ashps=df_ashps,
                                          T_out_ashp=T_out_ashp,
                                          T_amb=m.T_amb,
                                          tanks=tanks,
                                          df_tanks=df_tanks,
                                          indicator=Indicator,
                                          tank_costs = Tank_costs,
                                          df_boilers=df_boilers,
                                          boilers=boilers,
                                          KEEP_BOILER=KEEP_BOILER,
                                          KEEP_HPTK=KEEP_HPTK,
                                          )
        
        # Assigning the DES_MILP method in the full_model object to the Pyomo model m
        m = m.full_model.DES_MILP()
        
        # Deactivating the individual objectives in each block
        m.objective.deactivate()  
        
        # This is the free variable for total cost which the objective minimises
        m.cost = Var(bounds = (None, None))
        # m.cost = Var(bounds = (None, 7000)) #Octeract bound changes when this is included
        
        #This is the objective function rule that combines the costs for that particular season
        def rule_objective(m):
            expr = 0
            expr += (m.annual_cost_grid + m.carbon_cost + m.annual_inv_PV + \
                     m.annual_inv_HP + m.annual_inv_S + \
                         m.annual_oc_PV + m.annual_oc_b + m.annual_oc_S \
                             + m.annual_inv_B \
                             + m.annual_inv_Tank \
                             - m.export_income - m.gen_income)
            #expr += m.annual_cost_grid
            #expr += (sum(m.E_discharge[i,t,c] for i in m.i for t in m.t for c in m.c))
            return m.cost == expr
        m.obj_constr = Constraint(rule = rule_objective)
    
        
        # This function returns the model m to be used later within the code
        return m
    
    def DCOPF_block(m,s):
        # DC OPF for MILP
        m = opf_model.DCOPF()
        return m
    
    # Initialising the OPF class for DC approximation
    opf_model = DES_OPF(st=1, ft=FT,
                        I_MAX=None,
                        PHASE_SHIFT=PHASE_SHIFT,
                        primary_buses=PRIMARY_BUSES,
                        df_source=df_source,
                        slack=SLACK,
                        substation=SUBSTATION,
                        df_transformer=df_transformer,
                        df_lines=df_lines,
                        df_loads=df_loads,
                        df_buses=df_buses,
                        df_linecodes=df_linecodes,
                        )
    
    # Assigning the function to a Block so that it loops through all the seasons
    m.DES_res = Block(m.S, rule=seasonal_data)
    m.DCOPF = Block(m.S, rule=DCOPF_block)
    
    count = 0
    m.map_season_to_count = dict()
    m.first_season = None
    for i in m.S:
        m.map_season_to_count[count] = i
        if count == 0:
            m.first_season = i
        count += 1
    
    # Linking the capacities of all the DES technologies used within the model
    def linking_PV_panels_integers_rule(m,season,house,pnum):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].Y[house,pnum] == m.DES_res[previous_season].Y[house,pnum]
      
    m.PVLink1 = Constraint(m.S, houses, m.DES_res[1].v,
                                  rule = linking_PV_panels_integers_rule)
    
    def linking_PV_panels_reals_rule(m,season,house):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].panels_PV_reals[house] == m.DES_res[previous_season].panels_PV_reals[house]
      
    m.PVLink2 = Constraint(m.S, houses,
                                  rule = linking_PV_panels_reals_rule)
    
    def boiler_capacities_residential_rule(m,season,house,boiler):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].U[house, boiler] == m.DES_res[previous_season].U[house,boiler]
    
    m.boiler_linking_res = Constraint(m.S, houses, boilers, 
                                      rule = boiler_capacities_residential_rule)
    
    def battery_capacities_residential_rule(m,season,house,battery):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].W[house,battery] == m.DES_res[previous_season].W[house,battery]
    
    m.battery_linking_res = Constraint(m.S, houses, battery, 
                                       rule = battery_capacities_residential_rule)
    
    def hp_tank_capacities_residential_rule(m,season,house,ashp,tank):
        previous_season = None
        if season == m.first_season:
            return Constraint.Skip
        else:
            for key, val in m.map_season_to_count.items():
                if val == season:
                    previous_season = m.map_season_to_count[key-1]
                    return m.DES_res[season].J[house,ashp,tank] == m.DES_res[previous_season].J[house,ashp,tank]
    
    m.hp_tank_linking_res = Constraint(m.S, houses, ashps, tanks, 
                                       rule = hp_tank_capacities_residential_rule)
    
    # Ensure that powers here are given in p.u. by dividing with S_BASE
    def DC_P_linking(m,s,n,t):
        for load in houses:
            if load==n:
                return ((m.DES_res[s].E_PV_sold[n,t] - m.DES_res[s].E_grid[n,t] \
                    - sum(m.DES_res[s].E_grid_charge[n,t,b] for b in battery)))/S_BASE == m.DCOPF[s].P[n,t]
            else:
                continue
        else:
            return Constraint.Skip
    m.DC_active_power = Constraint(m.S, m.DCOPF[1].n, m.t, rule=DC_P_linking)
    
    ## Deactivating constraints related to complementarity
    for s in m.S:
        m.DES_res[s].SC9c.deactivate()
        m.DES_res[s].CP0a.deactivate()
        # m.DES_res[s].CP0b.deactivate()
        
    #This is the objective function combining residential costs for all seasons
    m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:]))
    
    return m


def NLP(m):
    for s in m.S:
        m.DCOPF[s].DC1.deactivate()
        m.DCOPF[s].DC2.deactivate()
        m.DCOPF[s].DC3.deactivate()
        m.DCOPF[s].DC4.deactivate()
        m.DCOPF[s].theta.unfix()
    m.DC_active_power.deactivate()
    
    # Fixing binary variables and capacities for the NLP initialisation
    for s in m.S:
        for h in m.DES_res[s].i:
            for v in m.DES_res[s].v:
                m.DES_res[s].Y[h,v].fix()
            for p in m.DES_res[s].p:
                for k in m.DES_res[s].k:
                    m.DES_res[s].J[h,p,k].fix()
            for b in m.DES_res[s].b:
                m.DES_res[s].U[h,b].fix()
            for t in m.DES_res[s].t:
                m.DES_res[s].X[h,t].fix()
                # Operation
                # m.DES_res[s].E_PV_sold[h,t].fix()
                for c in m.DES_res[s].c:
                    m.DES_res[s].Q[h,t,c].fix()
                    m.DES_res[s].W[h,c].fix()
                    # Operation
                    # m.DES_res[s].E_PV_charge[h,t,c].fix()   
                    
    
    ## Active power (kW) initialisation values for each season, conversion to p.u. happens in class
    power_init = {}
    for s in m.S:
        values = {}
        for n in houses:
            for t in m.t:
                values[n,t] = (m.DES_res[s].E_PV_sold[n,t].value \
                               - m.DES_res[s].E_grid[n,t].value \
                         - sum(m.DES_res[s].E_grid_charge[n,t,b].value \
                               for b in battery))
        power_init[s] = values
    # print(power_init)
    # print(" ")
    
    ## Loads (kW) in each season to initialise reactive power, conversion to p.u. happens in class
    load_init = {}
    for s in m.S:
        load_init[s] = m.DES_res[s].elec_house
        
    ## Creating the opf_model object from the DES_OPF class
    upf_object = UnbalancedPowerFlow(FT, SLACK, SLACK_ANGLES, PHASE_SHIFT, 
                                     df_source, df_buses, 
                                     df_loads, df_lines, 
                                      df_linecodes, 
                                     df_transformer,
                                     primary_buses=PRIMARY_BUSES,
                                     substation=SUBSTATION,
                                     # df_loadsinfo,
                                     )
           
    def OPF_block(m,s): 
        
        ## Assigning the OPF method in opf_model to the Pyomo model m
        m = upf_object.UPF(power_init=power_init[s], 
                           load_init=load_init[s])
        
        # This function also returns the Pyomo model 
        return m
    
    m.OPF_res = Block(m.S, rule=OPF_block)
    
    # Reactivating the nonlinear equations
    # for s in m.S:
    #     m.DES_res[s].INV6.activate()
        # deactivating the current constraint if required
        # m.OPF_res[s].C12.deactivate()
    
    ## Ensure that powers here are given in p.u. by dividing with S_BASE
    def P_linking(m,s,n,p,t):
        for load, phase in load_phase.items():
            if load==n and phase.lower()==p:
                return ((m.DES_res[s].E_PV_sold[n,t] - m.DES_res[s].E_grid[n,t] \
                    - sum(m.DES_res[s].E_grid_charge[n,t,b] for b in battery)))/S_BASE == m.OPF_res[s].P[n,p,t]
            elif load==n and phase.lower()!=p:
                return m.OPF_res[s].P[n,p,t] == 0
            else:
                continue
        else:
            return Constraint.Skip
    m.active_power = Constraint(m.S, m.OPF_res[1].n, m.OPF_res[1].p, m.t, rule=P_linking)
    # m.active_power.pprint()
        
    # TODO: Power factor and Q linking constraint
    def Q_linking(m,s,n,p,t):
        for load, phase in load_phase.items():
            if load==n and phase.lower()==p:
                # return m.OPF_res[s].Q[n,p,t] == 0
                return m.OPF_res[s].Q[n,p,t] == -m.DES_res[s].Q_load[load,t]/S_BASE
            elif load==n and phase.lower()!=p:
                return m.OPF_res[s].Q[n,p,t] == 0
            else:
                continue
        else:
            return Constraint.Skip
    m.reactive_power = Constraint(m.S, m.OPF_res[1].n, m.OPF_res[1].p, m.t, rule=Q_linking)
    
    return m

def VUB_calc(m, V_dict, theta_dict):
    VUB_dict = {}
    a = -0.5+0.866j
    for n in m.OPF_res[1].n:
        for t in range(1,FT+1):
            Va = V_dict[n,'a',t].value*(np.cos(theta_dict[n,'a',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'a',t].value)))
            Vb = V_dict[n,'b',t].value*(np.cos(theta_dict[n,'b',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'b',t].value)))
            Vc = V_dict[n,'c',t].value*(np.cos(theta_dict[n,'c',t].value) \
                                  + 1j*(np.sin(theta_dict[n,'c',t].value)))
            V1 = (Va+a*Vb+a**2*Vc)/3
            # print(V1)
            V2 = (Va+a**2*Vb+a*Vc)/3
            # print(V2)
            VUB_cmplx = V2/V1
            VUB = np.sqrt(np.real(VUB_cmplx)**2+np.imag(VUB_cmplx)**2)
            VUB_dict[n,t] = VUB*100
    return VUB_dict

# run_alg = True
start = perf_counter()

# import sys 
# stdoutOrigin=sys.stdout 
# sys.stdout = open("log.txt", "w")

def complementarity_algorithm(m, iteration, lowest_upper_bound, FAST_RESULTS):
    EPSILON = 0.1
    counter = 1
    NO_EARLY_TERMINATION = True
    FINAL_TOLERANCE = 1e-7
    while EPSILON >= FINAL_TOLERANCE and NO_EARLY_TERMINATION:
        print(f"\n------Iteration {counter}------\n")
        iter_start = perf_counter()
        for s in m.S:
            m.DES_res[s].EPSILON = EPSILON
        
        nlp_start_time = perf_counter()
        if SOLVER.lower() == 'conopt':
            solver = SolverFactory('gams')
            results = solver.solve(m, tee=True, solver = 'conopt',
                        add_options=['option reslim=100000;'])
        elif SOLVER.lower() == 'ipopt':
            solver = SolverFactory('ipopt')
            options = {}
            options['linear_solver'] = 'ma57'
            results = solver.solve(m, options = options, tee=True,)
            results = solver.solve(m, tee=True)
        else:
            print("this solver is unavailable")
        nlp_end_time = perf_counter()
        nlp_solver_times.append(nlp_end_time-nlp_start_time)
            
        # if counter == 1 and \
        #     results.solver.termination_condition == TerminationCondition.infeasible:
        #     print("\n **** Infeasible solution at relaxed complementarity ****")
        #     print("\n **** Terminating complementarity algorithm early ****")
        #     NO_EARLY_TERMINATION = False
        
        if FAST_RESULTS and counter >= 1 and \
        results.solver.termination_condition == TerminationCondition.locallyOptimal:
            # print("###### ", sum(m.DES_res[s].cost.value for s in m.S))
            # print("###### ", lowest_upper_bound)
            if sum(m.DES_res[s].cost.value for s in m.S) > lowest_upper_bound:
                print("\n **** Terminating complementarity algorithm early as TAC > LUB ****")
                NO_EARLY_TERMINATION = False
            
        if results.solver.termination_condition == TerminationCondition.locallyOptimal:
            EPSILON = EPSILON/10
        else:
            EPSILON = EPSILON*15
        
        prod_elec = {}
        prod_batt = {}
        CHECK_DICTS = 0
        for s in m.S:
            prod_elec[s]={k: m.DES_res[s].E_PV_sold[k].value*m.DES_res[s].E_grid[k].value for k, v in m.DES_res[s].E_PV_sold.items()}
            prod_batt[s]={k: m.DES_res[s].E_charge[k].value*m.DES_res[s].E_discharge[k].value for k, v in m.DES_res[s].E_charge.items()}
            if all(value >= FINAL_TOLERANCE for value in prod_elec[s].values())\
                and all(value >= FINAL_TOLERANCE for value in prod_batt[s].values()):
                    CHECK_DICTS += 0
            else:
                CHECK_DICTS +=1
        
        if CHECK_DICTS==0:
            NO_EARLY_TERMINATION=False
        
        iter_end = perf_counter()
        iter_comp_time = iter_end - iter_start
        # print(f"\n Iteration time: ", iter_comp_time)
        all_CR_iter_times.append(iter_comp_time)
        # EPSILON = EPSILON/10
        counter +=1
    
    with open(f'results_{mstr}_{size}_{tfind}.txt',"w") as f:
        f.write(str(results))

    for s in m.S:
        VUB[s] = VUB_calc(m, m.OPF_res[s].V, m.OPF_res[s].theta)


# Creating new Concrete Model per run
m = ConcreteModel()
# State the total amount of timestamps here:
m.t = RangeSet(FT) 
# Number of seasons, adjust the number accordingly:
m.S = RangeSet(SEASONS) 

binary_cut_pack = {}
LB_list = []
UB_list = []
iteration = 0
VUB = {}
all_CR_iter_times = []
nlp_solver_times = []
milp_solver_times = []

# Solve MILP
m = MILP(m)
deactivate_integer_design(m)

solver = SolverFactory('gams')
results = solver.solve(m, tee=True, 
                solver = 'cplex', add_options=['option optcr=0.000001;'])
with open(f'resultsCPLEX_{size}_{tfind}.txt',"w") as f:
    f.write(str(results))

lower_bound = m.obj.expr()
print(lower_bound)
LB_list.append(lower_bound)

if KEEP_OPF == 1:
    results_prints(m)

highest_lower_bound = lower_bound
lowest_upper_bound = 1e9
solution = 0
previous_upper_bound = 0

# Solve NLP
if KEEP_OPF == 1:
    m2 = m.clone()
    m2 = NLP(m2)
    if KEEP_COMPLEMENTARITY == 1:
    ##New deactivations and activations
        for s in m2.S:
            # Q constraints
            m2.DES_res[s].SC9a.deactivate()
            m2.DES_res[s].SC9b.deactivate()
            m2.DES_res[s].SC9c.activate()
            # X constraints
            # m.DES_res[s].bfg_constraint.deactivate()
            m2.DES_res[s].bbm_constraint.deactivate()
            m2.DES_res[s].stg_constraint.deactivate()
            m2.DES_res[s].CP0a.activate()
            # m.DES_res[s].CP0b.activate()
        
        complementarity_algorithm(m2, iteration, lowest_upper_bound, FAST_RESULTS)
        upper_bound = m2.obj.expr()
        print(upper_bound)
        UB_list.append(upper_bound)
        print(UB_list)
        solution = m2.clone()
        
        results_prints(m2)
        
        if upper_bound<lowest_upper_bound:
            lowest_upper_bound=upper_bound
        
        previous_upper_bound = upper_bound
            
        print(f'''\n The Highest Lower Bound = {highest_lower_bound} 
              \n The Lowest Upper Bound = {lowest_upper_bound}\n''')
        
        if BINARY_CUTS == 1:
            tic = perf_counter()
            binary_decisions = {}
            while round(lowest_upper_bound,4)>round(highest_lower_bound,4) and\
                perf_counter()<tic+TIMEOUT:   
                iteration += 1
                
                cut_pack_generator(m, iteration, binary_cut_pack)
                # with open(f'binary_cut_pack_{iteration}.py', 'w') as f:
                #     f.write(str(binary_cut_pack))
                
                cut_constraint_generator(m, iteration, binary_cut_pack,
                                KEEP_BATTERY, KEEP_PV, KEEP_BOILER, KEEP_HPTK,
                                FAST_RESULTS)
                print("Binary cuts successfully generated")
                
                milp_start_time = perf_counter()
                solver = SolverFactory('gams')
                results = solver.solve(m, tee=True, 
                                solver = 'cplex', add_options=['option optcr=0.00000001;'])
                milp_end_time = perf_counter()
                milp_solver_times.append(milp_end_time-milp_start_time)
                
                # Saving LB
                lower_bound = m.obj.expr()
                print(lower_bound)
                LB_list.append(lower_bound)
                print(LB_list)
                
                # results_prints(m)
                
                if lower_bound>highest_lower_bound:
                    highest_lower_bound=lower_bound
                else:
                    raise ValueError("Lower bound unstable! Terminating algorithm.\
                                     \Try tightening the MILP optimality gap")
                
                # Fix binary variables in m2 to those produced by m
                for s in m2.S:
                    for h in m2.DES_res[s].i:
                        for v in m2.DES_res[s].v:
                            m2.DES_res[s].Y[h,v].fix(m.DES_res[s].Y[h,v].value)
                        for b in m2.DES_res[s].b:
                            m2.DES_res[s].U[h,b].fix(m.DES_res[s].U[h,b].value)
                        for p in m2.DES_res[s].p:
                            for k in m.DES_res[s].k:
                                m2.DES_res[s].J[h,p,k].fix(m.DES_res[s].J[h,p,k].value)
                        for c in m2.DES_res[s].c:
                            m2.DES_res[s].W[h,c].fix(m.DES_res[s].W[h,c].value)
                
                # if FAST_RESULTS:
                #     if iteration >= 1:
                #         if iteration > 1:
                #             m2.del_component(m2.enforced)
                #         m2.enforced = Constraint(expr=sum(m2.DES_res[s].cost for s in m2.S) <= lowest_upper_bound)
                
                nlp_inits(m=m, m2=m2, load_phase=load_phase, S_BASE=S_BASE, SLACK=SLACK)
                complementarity_algorithm(m2, iteration, lowest_upper_bound, FAST_RESULTS)
                upper_bound = m2.obj.expr()
                print(upper_bound)
                UB_list.append(upper_bound)
                print(UB_list)
                
                if upper_bound<lowest_upper_bound:
                    lowest_upper_bound=upper_bound
                    results_prints(m2)
                    solution = m2.clone()
                
                print(f'''\n The Highest Lower Bound = {highest_lower_bound} 
                      \n The Lowest Upper Bound = {lowest_upper_bound}\n''')
                
                binary_decisions[iteration] = [{k: v.value for k,v in m.DES_res[1].J.items()},
                                                {k: v.value for k,v in m.DES_res[1].W.items()},
                                                ]
                # break         
    
            if perf_counter()>tic+TIMEOUT:
                print(f"\n *****************************************************")
                print(f"WARNING: The algorithm timed out before reaching convergence")
                print(f"The solution provided is the best available feasible solution from {iteration} iterations")
                print(f"\n *****************************************************")
    # TODO: Once m2 is solved and termination condition is met, make m2 = m for results output
    print(f"Final lower bound = {lower_bound}")
    m = solution
    
    
# Some prints to sanity check the costs
print(f"\n")
annual_grid = sum(m.annual_cost_grid.value for m in m.DES_res[:])
print("annual_grid_cost = " + str(annual_grid))
# day_grid = sum(m.cost_day.value for m in m.DES_res[:])
# print("day_cost = " + str(day_grid))
# night_grid = sum(m.cost_night.value for m in m.DES_res[:])
# print("night_cost = " + str(night_grid))
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

stop = perf_counter()
ex_time = stop - start 

print(f"\n****Total time*****: {ex_time}\n")

cost_dict = {'Objective':m.obj.expr(),
             "annual_grid":annual_grid,
             "carb_grid":carb_grid,
             "annual_PV_inv_cost":annual_PV_inv_cost,
             "annual_PV_op_cost":annual_PV_op_cost,
             "annual_boiler_inv_cost":annual_boiler_inv_cost,
             "annual_boiler_op_cost":annual_boiler_op_cost,
             "annual_batt_inv_cost":annual_batt_inv_cost,
             "annual_batt_op_cost":annual_batt_op_cost,
             "annual_ashp_inv_cost":annual_ashp_inv_cost,
             "annual_tank_inv_cost":annual_tank_inv_cost,
             "annual_export_inc":annual_inc,
             "annual_FIT":annual_FIT,
             "time taken":ex_time,
             'solver termination': results.solver.termination_condition,
             }

# sys.stdout.close()
# sys.stdout=stdoutOrigin

for s in m.S:
    rdf_cost = pd.DataFrame.from_dict(cost_dict, orient="index", columns=["value"])
    #for i,t in zip(house,m.t):
    '''residential results'''
    E_grid_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_grid.items()}
    rdf_result1 = pd.DataFrame.from_dict(E_grid_res_data, orient="index", columns=["variable value"])
    
    # panels_PV_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].panels_PV.items()}
    # rdf_result2 = pd.DataFrame.from_dict(panels_PV_res_data, orient="index", columns=["variable value"])
    
    E_PV_sold_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_sold.items()}
    rdf_result3 = pd.DataFrame.from_dict(E_PV_sold_res_data, orient="index", columns=["variable value"])
    
    E_PV_used_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_used.items()}
    rdf_result4 = pd.DataFrame.from_dict(E_PV_used_res_data, orient="index", columns=["variable value"])
    
    # Max_H_b_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].max_H_b.items()}
    # rdf_result5 = pd.DataFrame.from_dict(Max_H_b_res_data, orient="index", columns=["variable value"])
    
    # Storage_cap_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].storage_cap.items()}
    # rdf_result8 = pd.DataFrame.from_dict(Storage_cap_res_data, orient="index", columns=["variable value"])
    
    Q_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q.items()}
    rdf_result9 = pd.DataFrame.from_dict(Q_res_data, orient="index", columns=["variable value"])
    
    # Storage_volume_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].volume.items()}
    # rdf_result14 = pd.DataFrame.from_dict(Storage_volume_res, orient="index", columns=["variable value"])
    
    X_res_data = {(i, t, v.name): value(v) for (i,t), v in m.DES_res[s].X.items()}
    rdf_result16 = pd.DataFrame.from_dict(X_res_data, orient="index", columns=["variable value"])
    
    type_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].W.items()}
    rdf_result17 = pd.DataFrame.from_dict(type_res_data, orient="index", columns=["variable value"])
    
    # PV_num_data = {(i, r, v.name): r-1 for (i,r), v in m.DES_res[s].Y.items() if v!=0}
    # rdf_result20 = pd.DataFrame.from_dict(PV_num_data, orient="index", columns=["variable value"])
    
    PV_num_data = {(i, v.name): value(v) for i,v in m.DES_res[s].panels_PV_reals.items() if v!=0}
    rdf_result20 = pd.DataFrame.from_dict(PV_num_data, orient="index", columns=["variable value"])
    
    # HP_Z = {(i, p, v.name): value(v) for (i,p), v in m.DES_res[s].Z.items()}
    # rdf_hp1 = pd.DataFrame.from_dict(HP_Z, orient="index", columns=["variable value"])
    
    COP_data = {(t, p): value(v) for (t,p), v in m.DES_res[s].COP.items()}
    rdf_COP = pd.DataFrame.from_dict(COP_data, orient="index", columns=["variable value"])
    
    HP_Cap_data = {(t, p): value(v) for (t,p), v in m.DES_res[s].ashp_cap.items()}
    rdf_HPCap = pd.DataFrame.from_dict(HP_Cap_data, orient="index", columns=["variable value"])
    
    # Tank_J = {(i, k, v.name): value(v) for (i,k), v in m.DES_res[s].J.items()}
    # rdf_tk1 = pd.DataFrame.from_dict(Tank_J, orient="index", columns=["variable value"])
    Tank_J = {(i, p,k, v.name): value(v) for (i,p,k), v in m.DES_res[s].J.items()}
    rdf_tk1 = pd.DataFrame.from_dict(Tank_J, orient="index", columns=["variable value"])
    
    # Tank_cap_data = {(i, k, v.name): value(v) for (i,k), v in m.DES_res[s].H_max.items() if v!=0}
    # rdf_tk2 = pd.DataFrame.from_dict(Tank_cap_data, orient="index", columns=["variable value"])
    
    # New additions and OPF related exports
    # Inv_cap_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].inv_cap.items()}
    # rdf_result18 = pd.DataFrame.from_dict(Inv_cap_res, orient="index", columns=["variable value"])
    
    # PInv_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].P_inv.items()}
    # rdf_result19 = pd.DataFrame.from_dict(PInv_res_data, orient="index", columns=["variable value"])
    
    Boiler_U = {(i, b, v.name): value(v) for (i, b), v in m.DES_res[s].U.items()}
    rdf_blr1 = pd.DataFrame.from_dict(Boiler_U, orient="index", columns=["variable value"])
    
    Boiler_H = {(i, t, b, v.name): value(v) for (i, t, b), v in m.DES_res[s].H_b.items()}
    rdf_blr2 = pd.DataFrame.from_dict(Boiler_H, orient="index", columns=["variable value"])
    
    
    if KEEP_OPF == 1:
        Q_gen_res = {(i,t): v for (i,t), v in m.DES_res[s].Q_load.items()}
        rdf_result21 = pd.DataFrame.from_dict(Q_gen_res, orient="index", columns=["variable value"])
        
        V_res = {(n,p,t, v.name): value(v) for (n,p,t), v in m.OPF_res[s].V.items()}
        rdf_result22 = pd.DataFrame.from_dict(V_res, orient="index", columns=["variable value"])
        
        Theta_res = {(n,p,t, v.name): value(v) for (n,p,t), v in m.OPF_res[s].theta.items()}
        rdf_result23 = pd.DataFrame.from_dict(Theta_res, orient="index", columns=["variable value"])
        
        Q_res = {(n,p,t, v.name): value(v)*S_BASE for (n,p,t), v in m.OPF_res[s].Q.items()}
        rdf_result24 = pd.DataFrame.from_dict(Q_res, orient="index", columns=["variable value"])
        
        P_res = {(n,p,t, v.name): value(v)*S_BASE for (n,p,t), v in m.OPF_res[s].P.items()}
        rdf_result25 = pd.DataFrame.from_dict(P_res, orient="index", columns=["variable value"])
        
        # I_res = {(n,m,t, v.name): value(v) for (n,m,t),v in m.OPF_res[s].current_sqr.items()}
        # rdf_result26 = pd.DataFrame.from_dict(I_res, orient="index", columns=["variable value"])
        
        VUB_res = {(n,t): v for (n,t), v in VUB[s].items()}
        rdf_result27 = pd.DataFrame.from_dict(VUB_res, orient="index", columns=["variable value"])
    
    E_PVch_res = {}
    rdf_EPVch = {}
    E_stored_res = {}
    rdf_stored = {}
    E_gch_res = {}
    rdf_gridcharge = {}
    E_chg_res = {}
    rdf_chg = {}
    E_dsch_res = {}
    rdf_dsch = {}
    for bat_num in range(len(battery)):
        E_PVch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_PV_charge.items() if c == battery[bat_num]} 
        rdf_EPVch[bat_num] = pd.DataFrame.from_dict(E_PVch_res[bat_num], orient="index", columns=["variable value"])
        E_stored_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items() if c == battery[bat_num]}
        rdf_stored[bat_num] = pd.DataFrame.from_dict(E_stored_res[bat_num], orient="index", columns=["variable value"])
        E_gch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_grid_charge.items() if c == battery[bat_num]}
        rdf_gridcharge[bat_num] = pd.DataFrame.from_dict(E_gch_res[bat_num], orient="index", columns=["variable value"])
        E_chg_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_charge.items() if c == battery[bat_num]}
        rdf_chg[bat_num] = pd.DataFrame.from_dict(E_chg_res[bat_num], orient="index", columns=["variable value"])
        E_dsch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_discharge.items() if c == battery[bat_num]}
        rdf_dsch[bat_num] = pd.DataFrame.from_dict(E_dsch_res[bat_num], orient="index", columns=["variable value"])
    
    E_hp_res = {}
    rdf_E_hp = {}
    H_hp_res = {}
    rdf_H_hp = {}
    m_hp_res = {}
    rdf_m_hp = {}
    for hp_num in range(len(ashps)):
        E_hp_res[hp_num] = {(i, t, p, v.name): value(v) for (i,t,p), v in m.DES_res[s].E_hp.items() if p == ashps[hp_num]} 
        rdf_E_hp[hp_num] = pd.DataFrame.from_dict(E_hp_res[hp_num], orient="index", columns=["variable value"])
        H_hp_res[hp_num] = {(i, t, p,k, v.name): value(v) for (i,t,p,k), v in m.DES_res[s].H_hp.items() if p == ashps[hp_num]} 
        rdf_H_hp[hp_num] = pd.DataFrame.from_dict(H_hp_res[hp_num], orient="index", columns=["variable value"])
        # m_hp_res[hp_num] = {(i, t, p,k, v.name): value(v) for (i,t,p,k), v in m.DES_res[s].m_hp.items() if p == ashps[hp_num]} 
        # rdf_m_hp[hp_num] = pd.DataFrame.from_dict(m_hp_res[hp_num], orient="index", columns=["variable value"])
        
    H_store_res = {}
    rdf_H_store = {}
    H_ch_res = {}
    rdf_H_ch = {}
    H_disch_res = {}
    rdf_H_disch = {}
    H_loss_res = {}
    rdf_H_loss = {}
    H_unuse_res = {}
    rdf_H_unuse = {}
    T_tank_res = {}
    rdf_T_tank = {}
    for tk_num in range(len(tanks)):
        # H_store_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].H_stored.items() if k == tanks[tk_num]} 
        # rdf_H_store[tk_num] = pd.DataFrame.from_dict(H_store_res[tk_num], orient="index", columns=["variable value"])
        H_ch_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].H_charge.items() if k == tanks[tk_num]} 
        rdf_H_ch[tk_num] = pd.DataFrame.from_dict(H_ch_res[tk_num], orient="index", columns=["variable value"])
        H_disch_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].H_discharge.items() if k == tanks[tk_num]} 
        rdf_H_disch[tk_num] = pd.DataFrame.from_dict(H_disch_res[tk_num], orient="index", columns=["variable value"])
        H_loss_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].H_loss.items() if k == tanks[tk_num]} 
        rdf_H_loss[tk_num] = pd.DataFrame.from_dict(H_loss_res[tk_num], orient="index", columns=["variable value"])
        # H_unuse_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].H_unuse.items() if k == tanks[tk_num]} 
        # rdf_H_unuse[tk_num] = pd.DataFrame.from_dict(H_unuse_res[tk_num], orient="index", columns=["variable value"])
        T_tank_res[tk_num] = {(i, t, k, v.name): value(v) for (i,t,k), v in m.DES_res[s].T_tank.items() if k == tanks[tk_num]} 
        rdf_T_tank[tk_num] = pd.DataFrame.from_dict(T_tank_res[tk_num], orient="index", columns=["variable value"])
        
    Results_file_name1 = results_file_name+str(s)+results_file_suffix
    writer = ExcelWriter(Results_file_name1)
    rdf_cost.to_excel(writer, "Costs")
    rdf_result17.to_excel(writer, 'Battery_selection')
    rdf_result20.to_excel(writer, "PV_panels")
    # rdf_hp1.to_excel(writer, "ASHPs")
    rdf_COP.to_excel(writer, "COP")
    rdf_HPCap.to_excel(writer, "ASHP_Cap")
    rdf_tk1.to_excel(writer, "Tank_J")
    rdf_blr1.to_excel(writer, "Boiler_U")
    rdf_blr2.to_excel(writer, "Boiler_heat")
    # rdf_tk2.to_excel(writer, "Tank_Cap")
    # rdf_result8.to_excel(writer, 'Res_storage_cap')
    # rdf_result14.to_excel(writer, 'Res_stor_vol')
    # rdf_result2.to_excel(writer, 'Res_Panels_PV')
    # rdf_result5.to_excel(writer, 'Res_max_H_b')
    rdf_result1.to_excel(writer,'Res_E_grid')
    rdf_result3.to_excel(writer, 'Res_E_PV_sold')
    rdf_result4.to_excel(writer, 'Res_E_PV_used')
    
    for bat_num in range(len(battery)):
        for k, v in rdf_EPVch[bat_num].items():
            if sum(v) != 0:
                rdf_EPVch[bat_num].to_excel(writer, f"Res_E_PV_ch_{battery[bat_num]}")
        for k, v in rdf_stored[bat_num].items():
            if sum(v) != 0:
                rdf_stored[bat_num].to_excel(writer, f"Res_E_stored_{battery[bat_num]}")
        for k, v in rdf_gridcharge[bat_num].items():
            if sum(v) != 0:
                rdf_gridcharge[bat_num].to_excel(writer, f"Res_E_grd_ch_{battery[bat_num]}")
        for k, v in rdf_chg[bat_num].items():
            if sum(v) != 0:
                rdf_chg[bat_num].to_excel(writer, f"Res_E_charge_{battery[bat_num]}")
        for k, v in rdf_dsch[bat_num].items():
            if sum(v) != 0:
                rdf_dsch[bat_num].to_excel(writer, f"Res_E_disch_{battery[bat_num]}")
    
    for hp_num in range(len(ashps)):
        for k, v in rdf_E_hp[hp_num].items():
            if sum(v) != 0:
                rdf_E_hp[hp_num].to_excel(writer, f"E_hp_{ashps[hp_num]}")
        for k, v in rdf_H_hp[hp_num].items():
            if sum(v) != 0:
                rdf_H_hp[hp_num].to_excel(writer, f"H_hp_{ashps[hp_num]}")
        # for k, v in rdf_m_hp[hp_num].items():
        #     if sum(v) != 0:
        #         rdf_m_hp[hp_num].to_excel(writer, f"m_hp_{ashps[hp_num]}")
    
    for tk_num in range(len(tanks)):
        # for k, v in rdf_H_store[tk_num].items():
        #     if sum(v)!=0:
        #         rdf_H_store[tk_num].to_excel(writer, f"H_stor{tanks[tk_num]}")
        for k,v in rdf_H_ch[tk_num].items():
            if sum(v)!=0:
                rdf_H_ch[tk_num].to_excel(writer, f"H_ch{tanks[tk_num]}")
        for k,v in rdf_H_disch[tk_num].items():
            if sum(v)!=0:
                rdf_H_disch[tk_num].to_excel(writer, f"H_disch{tanks[tk_num]}")
        for k,v in rdf_H_loss[tk_num].items():
            if sum(v)!=0:
                rdf_H_loss[tk_num].to_excel(writer, f"H_loss{tanks[tk_num]}")
        # for k,v in rdf_H_unuse[tk_num].items():
        #     if sum(v)!=0:
        #         rdf_H_unuse[tk_num].to_excel(writer, f"H_unuse{tanks[tk_num]}")
        for k,v in rdf_T_tank[tk_num].items():
            if sum(v)!=0:
                rdf_T_tank[tk_num].to_excel(writer, f"T_tank{tanks[tk_num]}")
    
    if KEEP_OPF == 1:
        # rdf_result18.to_excel(writer, 'Inv_Cap')
        # rdf_result19.to_excel(writer, 'P_inv')
        rdf_result21.to_excel(writer, 'Q_gen')
        rdf_result22.to_excel(writer, 'V_OPF')
        rdf_result23.to_excel(writer, 'Angle_OPF')
        rdf_result25.to_excel(writer, 'P_node')
        rdf_result24.to_excel(writer, 'Q_node')
        # rdf_result26.to_excel(writer, 'I_sqr_node')
        rdf_result27.to_excel(writer, 'VUB')
        
    rdf_result9.to_excel(writer, 'Q')
    rdf_result16.to_excel(writer, 'X')
    
    writer.save()
    writer.close()


if KEEP_OPF == 1 and BINARY_CUTS == 1:
    writer = pd.ExcelWriter(bounds_file_name)
    bounds_df = pd.DataFrame({'LB':LB_list,'UB':UB_list})
    bounds_df.to_excel(writer,sheet_name='Bounds',index=True)
    nlp_times_df = pd.DataFrame({'NLP_Times': nlp_solver_times})
    nlp_times_df.to_excel(writer, sheet_name='nlp_times', index=True)
    milp_times_df = pd.DataFrame({'MILP_Times': milp_solver_times})
    milp_times_df.to_excel(writer, sheet_name='milp_times', index=True)
    cr_times_df = pd.DataFrame({'CR_Times': all_CR_iter_times})
    cr_times_df.to_excel(writer, sheet_name='CR_times', index=True)
    no_of_iter_df = pd.DataFrame({'Num': iteration},  index=[0])
    no_of_iter_df.to_excel(writer, sheet_name='iters', index=True)
    dfJ = pd.DataFrame([(k) for k,v in m.DES_res[1].J.items()], 
                       columns=['Bus', 'HP', 'Tank'])
    dfW = pd.DataFrame([(k) for k,v in m.DES_res[1].W.items()],
                       columns=['Bus', 'Battery'])
    for i, d in binary_decisions.items():
        dfJ[i]=d[0].values()
        dfW[i]=d[1].values()
    dfJ.to_excel(writer, sheet_name='J_bins', index=False)
    dfW.to_excel(writer, sheet_name='W_bins', index=False)
    writer.save()
    writer.close()
    
        
    
    
    

