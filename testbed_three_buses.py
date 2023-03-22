__author__ = ["David Cardona-Vasquez"]
__copyright__ = "Copyright 2022, Graz University of Technology"
__credits__ = ["David Cardona-Vasquez"]
__license__ = "MIT"
__maintainer__ = "David Cardona-Vasquez"
__status__ = "Development"

import numpy as np
import openpyxl
import os
import pathlib
import pyomo.environ as pyo
import pandas as pd
import yaml



def create_complete_model(input_data:dict):
    """
    This function creates the complete optimization model using the provided input data. Returns a Pyomo concrete model
    :param input_data:
    :return:
    """
    model = pyo.ConcreteModel(name="(Com)")

    the_gen = input_data['thermal_generators']
    wnd_gen = input_data['wind_generators']
    p = input_data['periods']
    pVC_g = input_data['vc_gen']
    pVC_nsp = input_data['vc_nsp']
    pCapFac_w = input_data['cf_wind']
    pMinProd_g = input_data['minprod']
    pMaxProd_g = input_data['maxprod']
    pDemand_p = input_data['demand']


    model.g = input_data['generators']
    model.t = pyo.Set(within=model.g, initialize=the_gen)
    model.w = pyo.Set(within=model.g, initialize=wnd_gen)
    model.p = pyo.RangeSet(p)

    model.pVC_g = pyo.Param(model.g, initialize=pVC_g)
    model.pVC_nsp = pyo.Param(initialize=pVC_nsp)
    model.pCapFac_w = pyo.Param(model.p, model.w, initialize=pCapFac_w, domain=pyo.NonNegativeReals)
    model.pMinProd_g = pyo.Param(model.g, initialize=pMinProd_g)
    model.pMaxProd_g = pyo.Param(model.g, initialize=pMaxProd_g)
    model.pDemand_p = pyo.Param(model.p, initialize=pDemand_p, domain=pyo.NonNegativeReals)

    model.vGen = pyo.Var(model.g, model.p, domain=pyo.NonNegativeReals)
    model.vNSP = pyo.Var(model.p, domain=pyo.NonNegativeReals)


    def eNSP_rule(mdl, i):
        return mdl.pDemand_p[i] >= mdl.vNSP[i]
    model.eNSP = pyo.Constraint(model.p, rule=eNSP_rule)

    def eBalance_rule(mdl, i):
        return mdl.pDemand_p[i] == sum(mdl.vGen[aux_g, i] for aux_g in mdl.g) + \
                mdl.vNSP[i]
    model.eBalance = pyo.Constraint(model.p, rule=eBalance_rule)

    def eMinProd_rule(mdl, g, i):
        return mdl.pMinProd_g[g] <= mdl.vGen[g, i]
    model.eMinProd = pyo.Constraint(model.g, model.p, rule=eMinProd_rule)

    def eMaxProd_rule(mdl, g, i):
        if g in mdl.w:
            return mdl.pMaxProd_g[g]*mdl.pCapFac_w[i, g] >= mdl.vGen[g, i]
        else:
            return mdl.pMaxProd_g[g] >= mdl.vGen[g, i]
    model.eMaxProd = pyo.Constraint(model.g, model.p, rule=eMaxProd_rule)

    def eCost_rule(mdl, i):
        return sum(mdl.vGen[g, i]*mdl.pVC_g[g] for g in mdl.g) + mdl.vNSP[i]*mdl.pVC_nsp
    model.vCost = pyo.Expression(model.p, rule=eCost_rule)

    def eObjective_rule(mdl):
        return sum(sum(mdl.vGen[g, i]*mdl.pVC_g[g] for g in mdl.g) + mdl.vNSP[i]*mdl.pVC_nsp for i in model.p)
    model.z = pyo.Objective(rule=eObjective_rule, sense=pyo.minimize)

    model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)


    return model


def create_complete_model_three_buses(input_data:dict):

    pLinLim_l = input_data['line_limit']

    model = create_complete_model(input_data)
    model.l = pyo.RangeSet(len(pLinLim_l))

    model.vLF = pyo.Var(model.l, model.l, model.p, domain=pyo.NonNegativeReals)
    model.pLinLim_l = pyo.Param(model.l, initialize=pLinLim_l)

    # delete single node balance and change it for balances in every node
    model.del_component('eBalance')
    def eBalance_bus_1(mdl, i):
        return mdl.pDemand_p[i] == - mdl.vLF[1, 3, i] - mdl.vLF[1, 2, i] + \
                mdl.vLF[3, 1, i] + mdl.vLF[2, 1, i] \
                + mdl.vNSP[i]
    model.eBalance_bus_1 = pyo.Constraint(model.p, rule=eBalance_bus_1)

    def eBalance_bus_2(mdl, i):
        return 0 == - mdl.vLF[2, 3, i] + mdl.vLF[1, 2, i] + \
               mdl.vLF[3, 2, i] - mdl.vLF[2, 1, i] + \
               mdl.vGen['t1', i]
    model.eBalance_bus_2 = pyo.Constraint(model.p, rule=eBalance_bus_2)

    def eBalance_bus_3(mdl, i):
        return 0 == mdl.vLF[2, 3, i] + mdl.vLF[1, 3, i] - \
               mdl.vLF[3, 1, i] - mdl.vLF[3, 2, i] + \
               mdl.vGen['w1', i]
    model.eBalance_bus_3 = pyo.Constraint(model.p, rule=eBalance_bus_3)

    # Constraints to represent the maximum flow
    def eMaxLim_1_exp_rule(mdl, i):
        return mdl.pLinLim_l[1] >= mdl.vLF[1, 2, i]
    model.eMaxLim_1_exp = pyo.Constraint(model.p, rule=eMaxLim_1_exp_rule)

    def eMaxLim_1_imp_rule(mdl, i):
        return mdl.pLinLim_l[1] >= mdl.vLF[2, 1, i]
    model.eMaxLim_1_imp = pyo.Constraint(model.p, rule=eMaxLim_1_imp_rule)

    def eMaxLim_2_exp_rule(mdl, i):
        return mdl.pLinLim_l[2] >= mdl.vLF[2, 3, i]
    model.eMaxLim_2_exp = pyo.Constraint(model.p, rule=eMaxLim_2_exp_rule)

    def eMaxLim_2_imp_rule(mdl, i):
        return mdl.pLinLim_l[2] >= mdl.vLF[3, 2, i]
    model.eMaxLim_2_imp = pyo.Constraint(model.p, rule=eMaxLim_2_imp_rule)

    def eMaxLim_3_exp_rule(mdl, i):
        return mdl.pLinLim_l[3] >= mdl.vLF[3, 1, i]
    model.eMaxLim_3_exp = pyo.Constraint(model.p, rule=eMaxLim_3_exp_rule)

    def eMaxLim_3_imp_rule(mdl, i):
        return mdl.pLinLim_l[3] >= mdl.vLF[1, 3, i]
    model.eMaxLim_3_imp = pyo.Constraint(model.p, rule=eMaxLim_3_imp_rule)


    # def eObjective_rule(mdl):
    #     return sum(sum(mdl.vGen[g, i]*mdl.pVC_g[g] for g in mdl.g) +
    #                mdl.vNSP[i]*mdl.pVC_nsp for i in model.p)
    # model.z = pyo.Objective(rule=eObjective_rule, sense=pyo.minimize)

    # model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model


def basis_summarization(sPath:str, basis:list) -> pd.DataFrame:
    """
    Receives a path to look for the basis and a list of the basis to consider;
    returns the input data (demand and capacity factor) for each basis
    :param sPath: folder with the Excel files with the basis are
    :param basis: string list with the basis to consider for the summarization
    :return: DataFrame with the demand and capacity factor of each basis
    """

    dfs = []
    for k in basis:
        auxPath = os.path.join(sPath, str(k)+".xlsx")
        wsDemand = pd.read_excel(auxPath, 5, index_col=0)
        wsCapFactors = pd.read_excel(auxPath, 6, index_col=0)

        df_aux = pd.merge(left=wsDemand, right=wsCapFactors, how='inner', left_index=True, right_index=True)
        df_aux.drop(columns=['generator'], inplace=True)
        df_aux['basis'] = str(k)

        dfs.append(df_aux.copy())

    dfAgg = pd.concat(dfs)
    dfAgg.reset_index(drop=True, inplace=True)

    return dfAgg


def data_load(sPath:str):
    """
    This function is used to load the data from the Excel files which have the parametrization of each 'basis' run
    of the power system
    :param file:
    :return:
    """

    wsConfig = pd.read_excel(sPath, 'config', header=None, index_col=0)
    wsThermal = pd.read_excel(sPath, 'thermal', index_col=0)
    wsVRES = pd.read_excel(sPath, 'vres', index_col=0)
    wsDemand = pd.read_excel(sPath, 'demand', index_col=0)
    wsCapFactors = pd.read_excel(sPath, 'cap_factors', index_col=0)

    input_data = dict()
    input_data['periods'] = wsConfig.loc['periods'].values[0]
    input_data['vc_nsp'] = wsConfig.loc['vc_nsp'].values[0]

    # get the data from the thermal generators
    the_gen = list(wsThermal.index)
    input_data['thermal_generators'] = the_gen

    # get the data from the wind generators
    wnd_gen = list(wsVRES.index)
    input_data['wind_generators'] = wnd_gen

    the_gen.extend(wnd_gen)
    input_data['generators'] = the_gen

    # use the data from thermal generators for its parameters
    input_data['maxprod'] = dict()
    input_data['minprod'] = dict()
    input_data['vc_gen'] = dict()
    for idx, r in wsThermal.iterrows():
        input_data['minprod'][idx] = r['min_cap']
        input_data['maxprod'][idx] = r['max_cap']
        input_data['vc_gen'][idx] = r['vc']

    # use the data from wind generators for its parameters
    for idx, r in wsVRES.iterrows():
        input_data['minprod'][idx] = r['min_cap']
        input_data['maxprod'][idx] = r['max_cap']
        input_data['vc_gen'][idx] = r['vc']

    # get data for capacity factors and demand
    input_data['demand'] = dict()
    input_data['cf_wind'] = dict()
    for idx, r in wsDemand.iterrows():
        input_data['demand'][idx] = r['demand']

    for idx, r in wsCapFactors.iterrows():
        input_data['cf_wind'][(idx, r['generator'])] = r['cap_factor']


    return input_data


def data_load_three_buses(sPath:str):
    """
    Loads only the required info for the three buses cases and calls the function
    to load the other information
    :param sPath:
    :return:
    """

    input_data = data_load(sPath)
    wsLineLimits = pd.read_excel(sPath, 'line_limits', index_col=0)

    input_data['line_limit'] = dict()
    for idx, r in wsLineLimits.iterrows():
        input_data['line_limit'][idx] = r['limit']


    return input_data


def basis_execution(folder:str, bases:dict, file='config.yaml', pRamping=False):
    """
    This function runs an optimization model considering the parametrized 'basis' in the yaml file. Each 'basis'must
    have an associated Excel file with the power system configuration to be used.
    :param folder:
    :return:
    """

    f = open(os.path.join(folder, file), 'r')
    d = yaml.safe_load(f)
    f.close()

    models = dict()
    for k, i in d['basis'].items():
        sPath = os.path.join(folder, k+'.xlsx')
        data = data_load_three_buses(sPath)
        model = create_complete_model_three_buses(data, pRamping)
        solver = pyo.SolverFactory("gurobi", solver_io="lp")
        res = solver.solve(model)
        aux_res = dict()

        # Fix later, checks if it is a simple basis and multiplies by weight
        # if 'bc' in k:
        #     aux_res['z'] = pyo.value(model.z) * bases[k]
        # else:
        #     aux_res['z'] = pyo.value(model.z)
        if not 'bc' in k:
            aux_res['z'] = pyo.value(model.z) * bases[k]
        else:
            aux_res['z'] = pyo.value(model.z)

        aux_res['model'] = model

        models[k] = aux_res

    return models


def run_complete_case(case='complete_1.xlsx', folder='data'):
    """
    This is used to run a complete model (without any aggregation) using the parametrization provided in the Excel file.
    The main differente with the 'basis' run is that it corresponds to only one complete run and thus it does not
    require any kind of weighting of the results.
    :return:
    """

    sPath = os.path.join(folder, case)
    data = data_load_three_buses(sPath)
    model = create_complete_model_three_buses(data)
    solver = pyo.SolverFactory("gurobi", solver_io="lp")
    res = solver.solve(model)

    return model

def get_rcs(n:str, rc:pd.DataFrame):
    values = []
    resources = []
    periods = []
    idx = rc.name == n
    for ix, r in rc.loc[idx].iterrows():
        periods.append(int(r['index'][1]))
        resources.append(str(n) + '_' + str(r['index'][0]))
        values.append(r['values'])

    aux_df = pd.DataFrame(list(zip(periods, resources, values)),
                          columns=['period', 'constraint', 'rc'])
    aux_df = aux_df.pivot(index='period', columns='constraint', values='rc').reset_index()

    return aux_df


def idx_match(row:pd.Series, df:pd.DataFrame):
    idxs = []
    for idx, r in df.iterrows():
        if all(row == r):
            idxs.append(idx)

    return idxs



def get_centroids(b_idx:dict, data:pd.DataFrame, cols=[0, 1], col_names=['demand', 'cap_factor']):

    b_cent = []
    names = []
    i=1
    for b in b_idx.values():
        aux = data.iloc[b, cols].apply(np.average, axis=0).values
        b_cent.append(aux)
        names.append('bs' + str(i))
        i = i + 1

    df_cent = pd.DataFrame(b_cent)
    df_cent.columns = col_names
    df_cent['basis'] = names

    return df_cent

def add_basis(b_idx:dict, data:pd.DataFrame):
    i=1
    data['basis'] = 'none'
    for b in b_idx.values():
        data.iloc[b, 2] = 'bs' + str(i)
        i = i + 1

def export_complete_solution(mdl: pyo.ConcreteModel, folder: str):

    sln = []

    if not os.path.exists(folder):
        os.mkdir(folder)

    # this code is only for illustrative purposes as it is not
    # the most efficient way to obtain data from Pyomo
    for v in mdl.component_objects(ctype=pyo.Var):
        if 'vLF' in str(v):
            continue
        aux_var = {'var': str(v)}
        aux_d = {}
        for idx in v:
            aux_d[idx] = pyo.value(v[idx])

        aux_var['values'] = aux_d

        sln.append(aux_var)

    for elem in sln:
        df = pd.DataFrame.from_dict(elem['values'], orient='index', columns=[str(elem['var'])])

        if not df.index.is_numeric():
            df.index = pd.MultiIndex.from_tuples(df.index)

        df.reset_index(drop=False, inplace=True)
        df.to_excel(os.path.join(folder, str(elem['var'])+'.xlsx'), index=False)


def export_aggregated_solution(mdls: dict, folder: str):

    # this code is only for illustrative purposes as it is not
    # the most efficient way to obtain data from Pyomo

    if not os.path.exists(folder):
        os.mkdir(folder)

    for k, v in mdls.items():
        mdl = v['model']
        export_complete_solution(mdl, os.path.join(folder, k))



def extract_duals(model, folder: str):

    aux_d = {}
    for k, v in model.dual.items():
        aux_name = k.parent_component().local_name
        if isinstance(k.index(), tuple):
            aux_name = aux_name + "_" + pyo.value(k.index()[0])

        try:
            aux_d[aux_name]
        except KeyError as e:
            aux_d[aux_name] = {}
        if isinstance(k.index(), tuple):
            aux_d[aux_name][pyo.value(k.index()[1])] = v
        else:
            aux_d[aux_name][pyo.value(k.index())] = v

    df_duals = pd.DataFrame.from_dict(aux_d['eBalance_bus_1'], orient='index').reset_index(drop=False)
    df_duals.rename(columns={0: 'eBalance_bus_1'}, inplace=True)
    df_temp = pd.DataFrame.from_dict(aux_d['eBalance_bus_2'], orient='index').reset_index(drop=True)
    df_duals = pd.merge(left=df_duals, right=df_temp, left_index=True, right_index=True, how='left')
    df_duals.rename(columns={0: 'eBalance_bus_2'}, inplace=True)
    df_temp = pd.DataFrame.from_dict(aux_d['eBalance_bus_3'], orient='index').reset_index(drop=True)
    df_duals = pd.merge(left=df_duals, right=df_temp, left_index=True, right_index=True, how='left')
    df_duals.rename(columns={0: 'eBalance_bus_3'}, inplace=True)

    for k, v in aux_d.items():
        if 'eBalance' not in str(k) :
            df_aux = pd.DataFrame.from_dict(aux_d[k], orient='index').reset_index(drop=False).rename(columns={0: k})
            df_duals = pd.merge(left=df_duals, right=df_aux, left_on='index', right_on='index', how='left')

    df_duals.rename(columns={'index': 'period'}, inplace=True)
    df_duals.fillna(0, inplace=True)

    if not os.path.exists(folder):
        os.mkdir(folder)

    df_duals.to_excel(os.path.join(folder, folder+"_duals.xlsx"), index=False)



if __name__ == '__main__':
    case = 'complete_three_buses.xlsx'
    # case = 'complete_3_three_buses_one_week.xlsx'
    model = run_complete_case(case)
    print('This is the main body, add code.')