import matplotlib.pyplot as plt
import matplotlib.cm as cm
from multiprocessing import Pool
import numpy as np                          # library used for linear algebra and vectorized operations
import openpyxl
import os
import pandas as pd
import pyomo.environ as pyo
import seaborn as sns
import testbed

from sklearn.cluster import KMeans          # sklearn is a library used for machine learning models

slvr = 'gurobi'     # change according to the solver installed in your machine
results_complete = testbed.run_complete_case(folder='data', case='complete_single_node.xlsx', solver=slvr)

df_complete, df_hourly = testbed.export_complete_solution(results_complete, 'complete_single_node')
df_complete

df_complete_duals = testbed.extract_duals(results_complete, 'complete_single_node')
df_complete_duals

df_cf = pd.read_excel(os.path.join('data', 'complete_single_node.xlsx'), sheet_name='cap_factors')
df_demand = pd.read_excel(os.path.join('data', 'complete_single_node.xlsx'), sheet_name='demand')

df_demand.head()
df_cf.head()

df_cf.drop(columns=['generator'], inplace=True)
df_cf.head()

df_input = pd.merge(left=df_demand, left_on='period', right=df_cf, right_on='period')
df_input.head()

df_input['demand_scaled'] = df_input['demand'] / df_input['demand'].max()

def cluster(i: int):
    clusterer = KMeans(n_clusters = i, random_state = 50, max_iter = 1000, n_init = 'auto')
    # clusterer = KMedoids(n_clusters = 3, random_state = 50, max_iter = 1000)


    df_input_clust = df_input.loc[:, ['cap_factor', 'demand_scaled']].copy()
    clusterer.fit(df_input_clust)

    df_input_clust["labels"] = clusterer.labels_ + 1

    df_centroids = pd.DataFrame(clusterer.cluster_centers_[:, 0:2])
    df_centroids.columns = ['cap_factor', 'demand']

    df_centroids['demand'] = df_centroids['demand'] * df_input['demand'].max()

    weights = df_input_clust.groupby('labels').count().reset_index(drop=True).loc[:, 'cap_factor']
    weights.rename('weight', inplace=True)
    df_centroids = pd.merge(left=df_centroids, right=weights, left_index=True, right_index=True)

    agg_path = 'aggregated_single_' + str(i) + '_clusters'
    path = os.path.join('data', agg_path)
    if not os.path.exists(path):
        os.makedirs(path)

    testbed.generate_config(df_centroids, folder=agg_path)



if __name__ == "__main__":

    lst_res = []
    for i in range(1, 8737):

        agg_path = 'aggregated_single_' + str(i) + '_clusters'

        if not os.path.exists(agg_path):
            os.makedirs(agg_path)

        cluster(i)
        results = testbed.basis_execution(folder=os.path.join('data', agg_path),
                                          solver=slvr, file='config_auto.xlsx')

        df_agg = testbed.export_aggregated_solution(results, agg_path)
        df_comparison = testbed.export_model_comparison(df_complete, df_agg)
        df_comparison['clusters'] = 'clusters_' + str(i)
        lst_res.append(df_comparison)
        cur_err = df_comparison.rel_delta[0]
        if cur_err <= 0.05:
            break

    df_final = pd.concat(lst_res)
    df_final.to_excel('df_comparison_final.xlsx', index=False)

    idx_of = df_final.result == 'of_value'
    x = range(1, int(df_final.shape[0]/5+1))
    plt.plot(x, df_final.rel_delta[idx_of])
    plt.xlim((0, 47))
    plt.ylabel('OF Error (%)')
    plt.xlabel('Number of Clusters (k-Means)')
    plt.hlines(y=cur_err, linestyles='dashed', colors='black', xmin=0, xmax=47)
    # plt.text(50, 50, 'error='+str(cur_err), fontsize=12)
    plt.text(5, 0.08, 'err=%.2f%%' % (cur_err*100), horizontalalignment='center',
         verticalalignment='center')
    plt.show()





