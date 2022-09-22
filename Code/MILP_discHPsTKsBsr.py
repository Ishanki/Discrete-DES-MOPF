from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from pandas import ExcelWriter

U_BOUND = None #100000
BIG_M = 200


class ResidentialSeasonalDES(object):
    def __init__(self, 
                 house, 
                 df, 
                 days, 
                 interval, 
                 ft,
                 irrad,
                 df_scalars, 
                 df_roof, 
                 elec_house, 
                 heat_house,
                 df_batteries,
                 # df_volume, 
                 battery, 
                 df_loads,
                 SEASONS,
                 KEEP_BATTERY,
                 KEEP_PV,
                 PV_OPTIONS,
                 ashps,
                 df_ashps,
                 T_out_ashp,
                 T_amb,
                 tanks,
                 df_tanks,
                 indicator,
                 tank_costs,
                 df_boilers,
                 boilers,
                 KEEP_BOILER,
                 KEEP_HPTK,
                 ):
        
        self.house = house #house names
        self.df = df #dataframe for electricity
        #self.dfh = dfh #dataframe for heating
        self.days = days #number of days in season
        self.interval = interval #duration of time interval
        self.ft = ft
        self.irrad = irrad #irradiance data
        self.df_scalars = df_scalars #dataframe with all the parameters
        self.df_roof = df_roof
        self.elec_house = elec_house
        self.heat_house = heat_house
        self.df_batteries = df_batteries
        # self.df_volume = df_volume
        self.battery = battery #battery types available
        self.df_loads = df_loads
        self.SEASONS=SEASONS
        self.KEEP_BATTERY=KEEP_BATTERY
        self.KEEP_PV=KEEP_PV
        self.PV_OPTIONS=PV_OPTIONS
        self.ashps=ashps
        self.df_ashps=df_ashps
        self.T_out_ashp=T_out_ashp
        self.T_amb=T_amb
        self.tanks = tanks
        self.df_tanks=df_tanks
        self.indicator=indicator
        self.tank_costs=tank_costs
        self.df_boilers=df_boilers
        self.boilers=boilers
        self.KEEP_BOILER=KEEP_BOILER
        self.KEEP_HPTK=KEEP_HPTK
        
    def DES_MILP(self):
        
        #constants extracted
        IRATE = self.df_scalars.iat[0,1]  # interest rate
        N_YEARS = self.df_scalars.iat[1,1]  #, doc = 'project lifetime')
        PRICE_GRID = self.df_scalars.iat[2,1]  #, doc = 'electricity price in £ per kWh')
        PRICE_GAS =self.df_scalars.iat[3,1]  #, doc = 'price of gas in £ per kWh')
        CARBON_GRID = self.df_scalars.iat[4,1]  #, doc = 'carbon intensity of grid electricity in kg/kWh')
        CC_PV = self.df_scalars.iat[5,1]  #, doc = 'capital cost of PV in £ per panel (1.75 m2)')
        N_PV = self.df_scalars.iat[6,1]  #, doc = 'efficiency of the PV')
        OC_FIXED_PV = self.df_scalars.iat[7,1]  #, doc = 'fixed operational cost of PV in £ per kW per year')
        OC_VAR_PV = self.df_scalars.iat[8,1]  #, doc = 'variable operational cost of PV in £ per kWh')
        TEX = self.df_scalars.iat[9,1]  #, doc = 'Tariff for exporting in £ per kWh')
        # CC_B = self.df_scalars.iat[10,1]  #, doc = 'capital cost of boiler per kWh')
        # N_B = self.df_scalars.iat[11,1]  #, doc = "thermal efficiency of the boiler")
        PANEL_AREA = self.df_scalars.iat[12,1]  #, doc = 'surface area per panel in m2')
        PANEL_CAPACITY = self.df_scalars.iat[13,1]  #, doc = 'rated capacity per panel in kW')
        MAX_CAPACITY_PV = self.df_scalars.iat[14,1]  #, doc = 'maximum renewables capacity the DES is allowed to have as per SEG tariffs')
        C_CARBON = self.df_scalars.iat[15,1]  #, doc = 'carbon cost for the grid')
        T_GEN = self.df_scalars.iat[16,1]  #, doc = 'generation tariff')
        PF = self.df_scalars.iat[17,1]  # doc = 'PV inverter power factor')
        N_INV = self.df_scalars.iat[18,1]  # doc = 'inverter efficiency')
        #GCA = self.df_scalars.iat[19,1]  # doc = 'states whether batteries can be charged from the grid or not'
        PG_NIGHT = self.df_scalars.iat[19,1]  # night tariff
        PG_DAY = self.df_scalars.iat[20,1]  # day tariff
        T_ROOM = self.df_scalars.iat[22,1] # set room temperature for heating
        
        model = ConcreteModel()
        
        model.i = Set(initialize= self.house, doc= 'residential users')
        model.periods2 = len(list(self.df[1])) #used to create a RangeSet for model.p
        model.t = RangeSet(model.periods2, doc= 'periods')
        model.t_night = RangeSet(1,7, doc = 'night periods eligible for night tariff')
        model.t_day = RangeSet(8,24, doc = 'day periods eligible for day tariff')
        model.c = Set(initialize = self.battery, doc='types of batteries available')
        model.v = Set(initialize=list(range(1,len(self.PV_OPTIONS)+1)))
        model.p = Set(initialize=self.ashps, doc='Air source heat pumps')
        model.k = Set(initialize=self.tanks, doc='Domestic Hot Water Tanks')
        model.b = Set(initialize=self.boilers, doc='Boilers')
        
        house_num = list(range(1,(len(self.house))+1))

        model.E_load = Param(model.i, model.t, initialize = self.elec_house, doc = 'electricity load')
        model.H_load = Param(model.i, model.t, initialize = self.heat_house, doc = 'heating load')
        #print(value(model.E_load['h1',34]))
        #print(value(model.H_load['h2',34]))
        
        ## Reactive power load calculation
        PF_dict = dict(zip(self.df_loads.Bus, self.df_loads.PF))
        # print(PF_dict)
        Q = {}
        for (k1,k2),v in model.E_load.items():
            Q[k1,k2] = sqrt((v**2)*((1/(PF_dict[k1]**2))-1))
        model.Q_load = Param(model.i, model.t, initialize=Q)
        # model.Q_load.pprint()
        
        PG = {}
        for t in model.t:
            if t in model.t_day:
                PG[t] = PG_DAY
            elif t in model.t_night:
                PG[t] = PG_NIGHT
        
        model.Irradiance = Param(model.t, initialize = self.irrad, doc = 'solar irradiance')
        model.Ta = Param(model.t, initialize=self.T_amb, doc='Ambient temperature')
        
        PV_panel_opt = {(h,i+1):i for h in model.i for i in self.PV_OPTIONS}
        # print(PV_panel_opt)
        model.panels_PV = Param(model.i, model.v, initialize=PV_panel_opt)
        
        ra = {}
        for n, h in zip(range(len(self.house)), self.house):
            ra[h] = self.df_roof.iat[n,1]
        model.max_roof_area = Param(model.i, initialize = ra, doc = 'maximum roof surface area available')
        
        '''battery parameters'''
        RTE1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            RTE1[c] = self.df_batteries.iat[n,1]
        #print(RTE1)
        model.RTE = Param(model.c, initialize = RTE1, doc='round trip efficiency')
        
        DCAP1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            DCAP1[c] = self.df_batteries.iat[n,2]
        # print(DCAP1)
        model.storage_cap = Param(model.c, initialize = DCAP1, doc= 'discrete capacity')
        
        mDoD1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            mDoD1[c] = self.df_batteries.iat[n,3]
        #print(mDoD1)
        
        model.max_DoD = Param(model.c, initialize = mDoD1, doc='max depth of discharge')

        mSoC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            mSoC1[c] = self.df_batteries.iat[n,4]
        #print(mSoC1)
        model.max_SoC = Param(model.c, initialize = mSoC1, doc='max state of charge')
        
        CCC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            CCC1[c] = self.df_batteries.iat[n,5]
        #print(CCC1)
        model.cc_storage = Param(model.c, initialize = CCC1, doc='cost per unit in £')
        
        OMC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            OMC1[c] = self.df_batteries.iat[n,6]
        #print(OMC1)
        model.om_storage = Param(model.c, initialize = OMC1, doc='operational and maintenance cost (£/y)')
        
        NEC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            NEC1[c] = self.df_batteries.iat[n,7]
        #print(NEC1)
        model.n_ch = Param(model.c, initialize = NEC1, doc='charging efficiency')
        
        NED1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            NED1[c] = self.df_batteries.iat[n,8]
        #print(NED1)
        model.n_disch = Param(model.c, initialize = NED1, doc='discharging efficiency')
        
        CCI1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            CCI1[c] = self.df_batteries.iat[n,9] 
        # print(CCI1)
        model.ci_storage = Param(model.c, initialize=CCI1, doc='installation cost')
        
        MCDP1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            MCDP1[c] = self.df_batteries.iat[n,10] 
        # print(MCDP1)
        model.max_cd_power = Param(model.c, initialize=MCDP1, doc='maximum charging and discharging power')
        
        L_cop = {}
        x0_cop = {}
        k_cop = {}
        b_cop = {}
        a_cap = {}
        b_cap = {}
        c_cap = {}
        d_cap = {}
        min_temp = {}
        ashp_uc = {}
        ashp_ic = {}
        min_flow = {}
        max_flow = {}
        for n, p in zip(range(len(self.ashps)), self.ashps):
            L_cop[p] = self.df_ashps.iat[n,2]
            x0_cop[p] = self.df_ashps.iat[n,3]
            k_cop[p] = self.df_ashps.iat[n,4]
            b_cop[p] = self.df_ashps.iat[n,5]
            a_cap[p] = self.df_ashps.iat[n,6]
            b_cap[p] = self.df_ashps.iat[n,7]
            c_cap[p] = self.df_ashps.iat[n,8]
            d_cap[p] = self.df_ashps.iat[n,9]
            min_temp[p] = self.df_ashps.iat[n,10]
            ashp_uc[p] = self.df_ashps.iat[n,11]
            ashp_ic[p] = self.df_ashps.iat[n,12]
            min_flow[p] = self.df_ashps.iat[n,13]/60    # converted to kg/s
            max_flow[p] = self.df_ashps.iat[n,14]/60    # converted to kg/s
        # print(x0_cop)
        # print(L_cap)
        # print(b0_hp)
        
        model.cc_ashp = Param(model.p, initialize=ashp_uc)
        model.ci_ashp = Param(model.p, initialize=ashp_ic)
        
        model.Tws = Param(initialize=self.T_out_ashp, doc='ASHP outlet temp')
        
        tnk_V = {}
        tnk_uc = {}
        tnk_n_ch = {}
        tnk_n_dch = {}
        T_min = {}
        tnk_radius = {}
        tnk_height = {}
        U_coeff = {}
        avg_heat_loss = {}
        for n, k in zip(range(len(self.tanks)), self.tanks):
            tnk_V[k] = self.df_tanks.iat[n,2]
            tnk_uc[k] = self.df_tanks.iat[n,3]
            tnk_n_ch[k] = self.df_tanks.iat[n,4]
            tnk_n_dch[k] = self.df_tanks.iat[n,5]
            T_min[k] = self.df_tanks.iat[n,6]
            tnk_radius[k] = self.df_tanks.iat[n,7]/2    # tank radius
            tnk_height[k] = self.df_tanks.iat[n,8]
            U_coeff[k] = self.df_tanks.iat[n,9]
            avg_heat_loss[k] = self.df_tanks.iat[n,10]
        model.V_tank = Param(model.k, initialize=tnk_V)
        model.cc_tank = Param(model.k, initialize=tnk_uc)
        model.n_ch_tank = Param(model.k, initialize=tnk_n_ch)
        model.n_disch_tank = Param(model.k, initialize=tnk_n_dch)
        
        def tank_surface_area(model,k):
            return (2*np.pi*tnk_radius[k]*tnk_height[k])+(2*np.pi*(tnk_radius[k]**2))
        model.A_tank = Param(model.k, initialize=tank_surface_area)
        
        def Loss_param_calc(model,k):
            return 0.0015*((model.V_tank[k])**-0.358)
        model.Loss_factor = Param(model.k, initialize=Loss_param_calc, 
                                  doc='empirical loss factor calculation')
        
        def COPfunc(model,t,p):
            return L_cop[p]/(1+np.exp(-k_cop[p]*(model.Ta[t]-x0_cop[p]))) + b_cop[p]
        model.COP = Param(model.t, model.p, initialize=COPfunc)
        # model.COP.pprint()
        
        def MaxCapfunc(model,t,p):
            return a_cap[p]*(model.Ta[t]**3) + b_cap[p]*(model.Ta[t]**2) + c_cap[p]*model.Ta[t] + d_cap[p]
        model.ashp_cap = Param(model.t, model.p, initialize=MaxCapfunc)
        # model.ashp_cap.pprint()
        
        Hmax_boiler = {}
        n_boiler = {}
        CC_boiler = {}
        CI_boiler = {}
        for n, b in zip(range(len(self.boilers)), self.boilers):
            Hmax_boiler[b] = self.df_boilers.iat[n,2]
            n_boiler[b] = self.df_boilers.iat[n,3]
            CC_boiler[b] = self.df_boilers.iat[n,4]
            CI_boiler[b] = self.df_boilers.iat[n,5]
        
        '''calculating capital recovery factor'''
        def capital_recovery_factor(model):
            return (IRATE * ((1 + IRATE)**N_YEARS))/(((1 + IRATE)**N_YEARS)-1)
        model.CRF = Param(initialize = capital_recovery_factor)
        print(f"this is CRF: {round(model.CRF.value,4)}")
        
        ## New parameter
        model.EPSILON = Param(initialize=0, mutable=True)
        
        # =============================================================================
        #               '''Variables and Initialisations'''
        # =============================================================================
        
        # Add initialisations to the variables + inverter variable declarations.
        
        '''Binary & integer variables'''
        model.X = Var(model.i, model.t, within=Binary, doc = '0 if electricity is bought from the grid')
        #model.YB = Var(model.i, within= Binary, doc = '1 if boiler is selected for the residential area')
        #model.Z = Var(model.i, within = Binary, doc = '1 if solar panels are selected for the residential area')
        model.Q = Var(model.i,model.t, model.c,  within = Binary, doc='1 if charging')
        # model.Q2 = Var(model.i,model.t, model.c, within = Binary, doc='1 if discharging')
        model.W = Var(model.i, model.c, within = Binary, doc = '1 if battery is selected')
        model.Y = Var(model.i, model.v, within=Binary, doc='1 if PV option selected')
        # model.Z = Var(model.i, model.p, within=Binary, doc='1 if ASHP option is selected',initialize=0)
        model.J = Var(model.i, model.p, model.k, within=Binary, doc='1 is tank option is chosen')
        # model.J = Var(model.i, model.k, within=Binary, doc='1 is tank option is chosen')
        model.U = Var(model.i, model.b, within=Binary, doc='1 if boiler is chosen')
        
        '''Positive variables'''
        model.panels_PV_reals = Var(model.i, within = NonNegativeReals, bounds = (0,5000), doc = 'number of panels of PVs installed')
        model.E_grid = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity imported from the grid in kW')
        model.E_PV_used = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity generated from the PV used at the house')
        model.E_PV_sold = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity generated from the PV sold to grid')
        model.H_b = Var(model.i, model.t, model.b, within = NonNegativeReals, doc = 'heat generated by boiler')
        model.annual_inv_PV = Var(within = NonNegativeReals, doc = 'investment cost of PV')
        model.annual_inv_B = Var(within = NonNegativeReals, doc = 'investment cost of boiler')
        model.annual_cost_grid = Var(within = NonNegativeReals, doc = 'cost of purchasing electricity from the grid')
        # model.cost_night = Var(within=NonNegativeReals, doc = 'annual cost of buying electricity during the night')
        # model.cost_day = Var(within=NonNegativeReals, doc = 'annual cost of buying electricity during the day')
        model.export_income= Var(within = NonNegativeReals, doc = 'Income from selling electricity from PVs to the grid')
        model.annual_oc_PV = Var(within = NonNegativeReals, doc = 'total opex of PVs')
        model.annual_oc_b = Var(within = NonNegativeReals, doc = 'total opex of boilers')
        #model.area_PV = Var(model.i, within = NonNegativeReals, doc = 'total PV area installed')
        model.max_H_b = Var(model.i, within = NonNegativeReals, doc = 'maximum heat generated by boilers')
        
        model.m_hp = Var(model.i, model.t, model.p, model.k, within=NonNegativeReals, doc='mass flow rate of hot water')
        model.E_hp = Var(model.i, model.t, model.p, within=NonNegativeReals, doc='ASHP electricity consumption in kW')
        model.H_hp = Var(model.i, model.t, model.p, model.k, within=NonNegativeReals, doc='ASHP heat production in kW')
        model.annual_inv_HP = Var(within=NonNegativeReals, doc='total investment cost of ASHPs')
        
        model.T_tank = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Uniform Tank Temperature')
        model.H_charge = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Total heat charged from heat source (ashps)')
        model.H_discharge = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Total heat discharged from tank')
        # model.H_stored = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Heat stored in tank in kWh')
        model.H_loss = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Heat lost from the tank in kWh')
        # model.H_max = Var(model.i, model.k, within=NonNegativeReals, doc='capacity of HW tank in kWh')
        # model.H_unuse = Var(model.i, model.t, model.k, within=NonNegativeReals, doc='Heat loss due to unuse')
        model.annual_inv_Tank = Var(within=NonNegativeReals, doc='total tank investment costs')

        # model.storage_cap = Var(model.i, model.c, within=NonNegativeReals, doc= 'installed battery capacity')
        model.E_stored = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity stored')
        model.E_grid_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity charged from grid')
        model.E_PV_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity charged from PVs')
        model.E_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'total electricity charged')
        model.E_discharge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'total electricity discharged from battery')
        # model.volume = Var(model.i, model.c, within=NonNegativeReals, doc = 'volume of battery installed')
        #model.E_disch_used = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.E_disch_sold = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.cycle = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'battery cycle number')
        model.annual_inv_S = Var(within = NonNegativeReals, doc = 'annual investment cost of batteries')
        model.annual_oc_S = Var(within = NonNegativeReals, doc = 'annual opex of batteries')
        
        model.carbon_cost = Var(within = NonNegativeReals, doc = 'carbon cost calculations')
        model.gen_income = Var(within = NonNegativeReals, doc = 'Total FIT generation income')
        
                
        # =============================================================================
        #               '''Constraints: General Electricity and Heating'''
        # =============================================================================
        '''Satisfying the demand with generated power'''
        def electricity_balance(model,i,t):
            return model.E_load[i,t] + sum(model.E_hp[i,t,p] for p in model.p) == model.E_grid[i,t] + model.E_PV_used[i,t] + (sum(model.E_discharge[i,t,c] for c in model.c)* self.KEEP_BATTERY) 
        model.eb_1 = Constraint(model.i, model.t, rule = electricity_balance, 
                                         doc = 'electricity consumed equals electricity generated')
        
        def heat_balance(model,i,t):
            return model.H_load[i,t] == (sum(model.H_discharge[i,t,k] for k in model.k)*self.KEEP_HPTK) + (sum(model.H_b[i,t,b] for b in model.b)*self.KEEP_BOILER)
        model.hb_1 = Constraint(model.i, model.t, rule = heat_balance, 
                                         doc = 'heat consumed = heat generated')
        
        # '''Ensuring that electricity generated is used first before sold to the grid '''
        # def buying_from_grid(model,i,t):
        #     return model.E_grid[i,t] <= model.E_load[i,t] * (1 - model.X[i,t])
        # model.bfg_constraint = Constraint(model.i, model.t, rule = buying_from_grid, 
        #                                   doc = 'electricity bought from grid <= electricity load * (1 - binary)')

        '''Ensuring that electricity generated is used first before sold to the grid '''
        def buying_from_grid(model,i,t):
            return model.E_grid[i,t] <= model.E_load[i,t] + sum(model.E_hp[i,t,p] for p in model.p)
        model.bfg_constraint = Constraint(model.i, model.t, rule = buying_from_grid, 
                                          doc = 'electricity bought from grid <= electricity load + heat pump consump.')
        
        def buying_bigM(model,i,t):
            return model.E_grid[i,t] <= BIG_M*(1 - model.X[i,t])
        model.bbm_constraint = Constraint(model.i, model.t, rule=buying_bigM,
                                          doc='new linearisation')
        
        def selling_to_grid(model,i,t):
            return model.E_PV_sold[i,t] <= BIG_M * model.X[i,t]
        model.stg_constraint = Constraint(model.i, model.t, rule = selling_to_grid, 
                                          doc = 'electricity sold <=  some upper bound * binary')
        
        def complementarity_X(model,i,t):
            return model.E_PV_sold[i,t]*model.E_grid[i,t] <= model.EPSILON
        model.CP0a = Constraint(model.i, model.t, rule=complementarity_X)
        
        # def bfg2(model,i,t):
        #     return model.E_grid[i,t] <= model.E_load[i,t] + sum(model.E_hp[i,t,p] for p in model.p)
        # model.CP0b = Constraint(model.i, model.t, rule=bfg2)
                
        # =============================================================================
        #                           '''Constraints: PVs'''
        # =============================================================================
        '''Power balance for PVs'''
        def PV_binary(model,i):
            return sum(model.Y[i,v] for v in model.v) <= 1
        model.PVN1 = Constraint(model.i, rule=PV_binary)
        
        def PV_generation(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * PANEL_AREA * model.Irradiance[t] * N_PV * self.KEEP_PV
        model.PVN2 = Constraint(model.i, model.t, rule = PV_generation, 
                                          doc = 'total electricity generated by PV <= PV area * Irradiance * PV efficiency')
        
        '''Electricity generated by PVs not exceeding rated capacity'''
        def PV_rated_capacity(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * PANEL_CAPACITY
        model.PVN3 = Constraint(model.i, model.t, rule = PV_rated_capacity, 
                                          doc = 'total electricity generated by PV <= installed PV area * capacity of each panel/surface area of each panel')
        
        '''Investment cost of PVs per year'''
        def PV_investment(model):
            return (sum(CC_PV * sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * model.CRF for i in model.i))/self.SEASONS == model.annual_inv_PV
        model.PVN4 = Constraint(rule = PV_investment, 
                                          doc = 'sum for all residences(capital cost * surface area/1.75 * CRF, N.B. 1.75 is the surface area per panel')
        
        '''Operation and maintenance cost of PVs per year'''
        def PV_operation_cost(model):
            return sum((sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * OC_FIXED_PV * (1/365) * self.days * PANEL_CAPACITY) for i in model.i) == model.annual_oc_PV
        model.PVN5 = Constraint(rule = PV_operation_cost, doc = 'sum of variable and fixed costs')        
        
        '''Roof area limitation'''
        def maximum_roof_area(model,i):
            return sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * PANEL_AREA <= model.max_roof_area[i]
        model.PVN6 = Constraint(model.i, rule = maximum_roof_area, 
                                          doc = 'total PV area installed cannot exceed max roof area at each residence')
        
        '''Capacity limitation imposed by Smart Export Guarantee'''
        def SEG_capacity_limitation(model):
            return sum(sum(model.panels_PV[i,v]*model.Y[i,v] for v in model.v) * PANEL_CAPACITY for i in model.i) <= MAX_CAPACITY_PV
        model.PVN7 = Constraint(rule = SEG_capacity_limitation, 
                                            doc = 'sum of PVs installed in all houses cannot exceed maximum capacity given by the tariff regulations')
        
        ############# '''Relaxed formulation''' #################
        '''Power balance for PVs'''
        def PV_generation(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= model.panels_PV_reals[i] * PANEL_AREA * model.Irradiance[t] * N_PV * self.KEEP_PV
        model.PVR1 = Constraint(model.i, model.t, rule = PV_generation, 
                                          doc = 'total electricity generated by PV <= PV area * Irradiance * PV efficiency')
        
        '''Electricity generated by PVs not exceeding rated capacity'''
        def PV_rated_capacity(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= model.panels_PV_reals[i] * PANEL_CAPACITY
        model.PVR2 = Constraint(model.i, model.t, rule = PV_rated_capacity, 
                                          doc = 'total electricity generated by PV <= installed PV area * capacity of each panel/surface area of each panel')
        
        '''Investment cost of PVs per year'''
        def PV_investment(model):
            return (sum(CC_PV * model.panels_PV_reals[i] * model.CRF for i in model.i))/self.SEASONS == model.annual_inv_PV
        model.PVR3 = Constraint(rule = PV_investment, 
                                          doc = 'sum for all residences(capital cost * surface area/1.75 * CRF, N.B. 1.75 is the surface area per panel')
        
        '''Operation and maintenance cost of PVs per year'''
        def PV_operation_cost(model):
            return sum((model.panels_PV_reals[i] * OC_FIXED_PV * (1/365) * self.days * PANEL_CAPACITY) for i in model.i) == model.annual_oc_PV
        model.PVR4 = Constraint(rule = PV_operation_cost, doc = 'sum of variable and fixed costs')        
        
        '''Roof area limitation'''
        def maximum_roof_area(model,i):
            return model.panels_PV_reals[i] * PANEL_AREA <= model.max_roof_area[i]
        model.PVR5 = Constraint(model.i, rule = maximum_roof_area, 
                                          doc = 'total PV area installed cannot exceed max roof area at each residence')
        
        '''Capacity limitation imposed by Smart Export Guarantee'''
        def SEG_capacity_limitation(model):
            return sum(model.panels_PV_reals[i] * PANEL_CAPACITY for i in model.i) <= MAX_CAPACITY_PV
        model.PVR6 = Constraint(rule = SEG_capacity_limitation, 
                                            doc = 'sum of PVs installed in all houses cannot exceed maximum capacity given by the tariff regulations')
        
        # =============================================================================
        #                       '''Constraints: ASHPs'''
        # =============================================================================
        
        '''Power and Heat relationship'''
        def heat_power(model,i,t,p):
            return model.E_hp[i,t,p]==sum(model.H_hp[i,t,p,k] for k in model.k)/model.COP[t,p]
        model.HPC1 = Constraint(model.i, model.t, model.p, rule=heat_power)
        
        '''Maximum heat production'''
        def max_heat_hp(model,i,t,p):
            return sum(model.H_hp[i,t,p,k] for k in model.k) <= model.ashp_cap[t,p]*sum(model.J[i,p,k] for k in model.k)
        model.HPC2 = Constraint(model.i, model.t, model.p, rule=max_heat_hp)
        
        '''Temperature constraint'''
        def min_op_temp(model,i,t,p):
            if model.Ta[t] <= min_temp[p]:
                return sum(model.H_hp[i,t,p,k] for k in model.k) == 0
            else:
                return Constraint.Skip
        model.HPC3 = Constraint(model.i, model.t, model.p, rule=min_op_temp)
        
        '''Investment costs'''
        def ashp_investment(model):
             return sum(sum(sum(model.J[i,p,k] for k in model.k) * (model.cc_ashp[p]+model.ci_ashp[p]) for p in model.p) * model.CRF/self.SEASONS for i in model.i) == model.annual_inv_HP
        model.HPC4 = Constraint(rule = ashp_investment)
        
        # =============================================================================
        #                       '''Constraints: HW Tanks'''
        # =============================================================================
        
        '''Tank option restriction'''
        def tank_binary(model,i):
            return sum(model.J[i,p,k] for p in model.p for k in model.k) <= 1
        model.TK1 = Constraint(model.i, rule=tank_binary)
        
        '''HP-Tank Combination restriction'''
        def hp_tank_combo(model,i,p,k):
            return model.J[i,p,k] <= self.indicator[p,k]
        model.TK1b = Constraint(model.i, model.p, model.k, rule=hp_tank_combo)
        
        '''Relationship between heat charging variable and heat output from ashp'''
        def total_heat_charge(model,i,t,k):
            return sum(model.H_hp[i,t,p,k] for p in model.p) == model.H_charge[i,t,k]
        model.TK2 = Constraint(model.i, model.t, model.k, rule=total_heat_charge)
        
        '''Heat storage balance'''
        def heat_storage_balance(model,i,t,k):
            if t>1:
                return (4.182*model.V_tank[k]*1000*(model.T_tank[i,t,k]-model.T_tank[i,t-1,k])/(self.interval*3600)) == (model.H_charge[i,t,k]*model.n_ch_tank[k]) - (model.H_discharge[i,t,k]/model.n_disch_tank[k]) - model.H_loss[i,t,k]
            else:
                return (4.182*model.V_tank[k]*1000*(model.T_tank[i,t,k]-model.Ta[t]*sum(model.J[i,p,k] for p in model.p))/(self.interval*3600)) == (model.H_charge[i,t,k]*model.n_ch_tank[k]) - (model.H_discharge[i,t,k]/model.n_disch_tank[k]) - model.H_loss[i,t,k]
        model.TK3 = Constraint(model.i, model.t, model.k, rule=heat_storage_balance)
        
        '''Heat loss equation'''
        def heat_loss_overall(model,i,t,k):
            return model.H_loss[i,t,k] == avg_heat_loss[k]*sum(model.J[i,p,k] for p in model.p)
        model.TK4a = Constraint(model.i, model.t, model.k, rule=heat_loss_overall)
        
        def T_tank_LB(model,i,t,k):
            return model.T_tank[i,t,k] >= T_min[k]*sum(model.J[i,p,k] for p in model.p)
        #model.T_tank[i,t,k] >= T_min[k]*model.J[i,k]
        model.TK4b1 = Constraint(model.i, model.t, model.k, rule=T_tank_LB)
        
        def T_tank_UB(model,i,t,k):
            return model.T_tank[i,t,k] <= model.Tws*sum(model.J[i,p,k] for p in model.p)
        #model.T_tank[i,t,k] <= model.Tws*model.J[i,k]
        model.TK4b2 = Constraint(model.i, model.t, model.k, rule=T_tank_UB)
        
        '''Linking starting and ending storage levels'''
        def heat_store_start_end(model,i,k):
            return model.T_tank[i,1,k] == model.T_tank[i,self.ft,k]
        model.TK7 = Constraint(model.i, model.k, rule=heat_store_start_end)
        
        def trouble1(model,i,t,k):
            return model.H_charge[i,t,k] <= BIG_M*sum(model.J[i,p,k] for p in model.p)
        model.TB1 = Constraint(model.i, model.t, model.k, rule=trouble1)
        
        def trouble2(model,i,t,k):
            return model.H_discharge[i,t,k] <= BIG_M*sum(model.J[i,p,k] for p in model.p)
        model.TB2 = Constraint(model.i, model.t, model.k, rule=trouble2)
        
        '''Capital cost'''
        def capex_tank(model):
            return model.annual_inv_Tank == sum(sum(sum(model.J[i,p,k]*self.tank_costs[p,k] for p in model.p) for k in model.k) * model.CRF/self.SEASONS for i in model.i)
        model.TK9 = Constraint(rule=capex_tank)
        
        # =============================================================================
        #                       '''Constraints: Boilers'''
        # =============================================================================
        '''Only 1 boiler allowed'''
        def boiler_binary(model,i):
            return sum(model.U[i,b] for b in model.b) <= 1
        model.BC1 = Constraint(model.i, rule=boiler_binary)
        
        '''Maximum heat generated'''
        def max_heat_boiler(model,i,t,b):
            return model.H_b[i,t,b] <= Hmax_boiler[b]*model.U[i,b]
        model.BC2 = Constraint(model.i, model.t, model.b, rule=max_heat_boiler)
        
        '''Investment cost of boiler per year'''
        def boiler_investment(model):
            return (sum(sum(model.U[i,b]*(CC_boiler[b]+CI_boiler[b]) for b in model.b) * model.CRF for i in model.i))/self.SEASONS == model.annual_inv_B
        model.bi_constraint = Constraint(rule = boiler_investment,
                                          doc = 'investment cost = sum for all residences(capital cost * boiler capacity * CRF)') 
        
        '''Operation and maintenance cost of boilers per year'''
        def boiler_operation_cost(model):
            return sum(model.H_b[i,t,b] * self.interval * (PRICE_GAS/n_boiler[b]) * self.days  for i in model.i for t in model.t for b in model.b) == model.annual_oc_b
        model.boc_constraint = Constraint(rule = boiler_operation_cost,
                                          doc = 'for all residences, seasons and periods, the heat generated by boiler * fuel price/thermal efficiency * no.of days' )
        
        # =============================================================================
        #                       '''Constraints: Batteries'''
        # =============================================================================
        '''Only 1 type of battery can be installed at each house'''
        def battery_type(model,i):
            return sum(model.W[i,c] for c in model.c) <= 1 * self.KEEP_BATTERY
        model.SC0 = Constraint(model.i, rule = battery_type)
        
        '''Maximum charging and discharging power'''
        def max_charging_power(model,i,t,c):
            return model.E_charge[i,t,c] <= model.max_cd_power[c]*model.W[i,c]
        model.SC1a = Constraint(model.i, model.t, model.c, rule = max_charging_power,
                               doc = 'charging power limited to max')
        
        def max_discharging_power(model,i,t,c):
            return model.E_discharge[i,t,c] <= model.max_cd_power[c]*model.W[i,c]
        model.SC1b = Constraint(model.i, model.t, model.c, rule = max_discharging_power,
                               doc = 'discharging power limited to max')

        '''Maximum SoC limitation'''
        def battery_capacity1(model,i,t,c):
            return model.E_stored[i,t,c] <= model.storage_cap[c] * model.max_SoC[c] * model.W[i,c]
        model.SC2a = Constraint(model.i, model.t, model.c, rule = battery_capacity1,
                                         doc = 'the energy in the storage cannot exceed its capacity based on the volume available and maximum state of charge')
 
        '''Maximum DoD limitation'''
        def battery_capacity2(model,i,t,c):
            return model.E_stored[i,t,c] >= model.storage_cap[c] * (1-model.max_DoD[c]) * model.W[i,c]
        model.SC2b = Constraint(model.i, model.t, model.c, rule = battery_capacity2,
                                         doc = 'the energy in the storage has to be greater than or equal to its capacity based on the volume available and maximum depth of discharge')
        
        '''Battery storage balance'''
        def storage_balance(model,i,t,c):
            if t > 1:
                return model.E_stored[i,t,c] == model.E_stored[i,t-1,c] + (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            else:
                return model.E_stored[i,t,c] == (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            #Constraint.Skip
        model.SC3 = Constraint(model.i, model.t, model.c, rule = storage_balance,
                                         doc = 'Energy stored at the beginning of each time interval is equal unused energy stored + energy coming in - Energy discharged')
        
        def fixing_start_and_end1(model,i,t,c):
            return model.E_stored[i,1,c] == model.E_stored[i,self.ft,c]
        model.SC5 = Constraint(model.i, model.t, model.c, rule = fixing_start_and_end1,
                                  doc = 'battery capacity at the start and end of the time horizon must be the same')
        
        '''Total electricity used to charge battery'''
        def total_charge(model,i,t,c):
            return model.E_charge[i,t,c] ==  model.E_grid_charge[i,t,c] + model.E_PV_charge[i,t,c]
        model.SC6 = Constraint(model.i, model.t, model.c, rule = total_charge,
                                         doc = 'total charging power = charging electricity from PVs + grid')

        '''Investment Cost of Batteries per year'''
        def storage_investment(model):
             return sum(sum(model.W[i,c] * (model.cc_storage[c]+model.ci_storage[c]) for c in model.c) * model.CRF/self.SEASONS for i in model.i) == model.annual_inv_S
        model.SC7 = Constraint(rule = storage_investment,
                                          doc = 'investment cost = capacity (kWh) * cost per kWh * CRF')

        '''Operational and Maintenance cost of batteries per year'''
        def storage_operation_cost(model):
            return sum(sum(model.W[i,c] * model.om_storage[c] for c in model.c) * (1/365) * self.days for i in model.i) == model.annual_oc_S
        model.SC8 = Constraint(rule = storage_operation_cost,
                                         doc = 'opex = capacity (kWh) * cost per kWh per year')
        
        '''Ensuring that battery cannot charge and discharge at same time'''
        def charging_bigM(model,i,t,c):
            return model.E_charge[i,t,c] <= BIG_M * model.Q[i,t,c]
        model.SC9a = Constraint(model.i, model.t, model.c,rule = charging_bigM,
                               doc = 'Q will be 1 if charging')
        
        def discharging_bigM(model,i,t,c):
            return model.E_discharge[i,t,c] <= BIG_M * (1-model.Q[i,t,c])
        model.SC9b = Constraint(model.i, model.t, model.c,rule = discharging_bigM,
                               doc = 'Q2 will be 1 if discharging')
        
        def complementarity_Q(model,i,t,c):
            return model.E_charge[i,t,c]*model.E_discharge[i,t,c] <= model.EPSILON
        model.SC9c = Constraint(model.i, model.t, model.c, rule=complementarity_Q)
        
        # =============================================================================
        #                    '''Constraints: General Costs'''
        # =============================================================================
        
        '''total cost of buying electricity during the day'''
        def grid_cost(model):
            return model.annual_cost_grid == sum(((model.E_grid[i,t] + sum(model.E_grid_charge[i,t,c] for c in model.c)) * self.interval * PG[t] * self.days) for i in model.i for t in model.t)
        model.GC = Constraint(rule = grid_cost)
        
        '''carbon cost for the grid'''
        def seasonal_carbon_cost(model):
            return model.carbon_cost == sum(((model.E_grid[i,t] + sum(model.E_grid_charge[i,t,c] for c in model.c)) * self.interval * CARBON_GRID * self.days) for i in model.i for t in model.t) * C_CARBON
        model.GC2 = Constraint(rule = seasonal_carbon_cost)
        
        '''Income from selling electricity to the grid'''
        def income_electricity_sold(model):
            return model.export_income == sum((model.E_PV_sold[i,t] * self.interval * TEX * self.days) for i in model.i for t in model.t)
        model.ies_constraint = Constraint(rule = income_electricity_sold,
                                         doc = 'income = sum for all residences, seasons and periods(electricity sold * smart export guarantee tariff')
        
        '''FIT generation'''
        def FIT(model):
            return model.gen_income == sum(((model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c))* self.interval * T_GEN * self.days) for i in model.i for t in model.t)
        model.I1 = Constraint(rule = FIT)
        
        def objective_rule(model):
            return (model.annual_cost_grid + model.carbon_cost \
                    + model.annual_inv_PV + model.annual_inv_HP \
                    + model.annual_inv_S + model.annual_oc_PV \
                    + model.annual_oc_b + model.annual_oc_S \
                    + model.annual_inv_B \
                    + model.annual_inv_Tank \
                    - model.export_income - model.gen_income)
        model.objective = Objective(rule= objective_rule, sense = minimize, doc = 'objective function')


        return model