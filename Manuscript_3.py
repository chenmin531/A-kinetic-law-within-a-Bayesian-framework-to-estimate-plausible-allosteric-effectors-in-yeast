import os
import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import theano
import theano.tensor as tt
import math
from scipy.stats.distributions import chi2
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from collections import OrderedDict
import shutil
from scipy import optimize
from theano.compile.ops import as_op
mpl.use('TkAgg')
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




def process_enzyme_id_to_df(enzyme_id):
    cleaned_id = re.sub(r'\(|\)', '', enzyme_id)
    cleaned_id = re.sub(r' AND | OR ', ' ', cleaned_id)
    enzyme_list = cleaned_id.split()
    unique_enzymes = list(set(enzyme_list))
    unique_enzymes.sort()
    enzyme_df = pd.DataFrame(unique_enzymes, columns=['Enzyme_ID'])
    enzyme_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
    enzyme_df.insert(loc=2, column='Measured_value', value=reaction_ID)
    for enz in range(0, len(enzyme_df)):
        enz_involved = enzyme_df.iloc[enz,1]
        enz_screened_name = np.delete(
            np.mat(Transcripts .loc[(Transcripts [list(Transcripts )[0]] == enz_involved)]), 0, 1)
        if pd.isnull(list(enz_screened_name)).all():
            eny_measured_value = "False"
            enzyme_df.iloc[enz, 2] = eny_measured_value
        else:
            eny_measured_value = "True"
            enzyme_df.iloc[enz, 2] = eny_measured_value
    return enzyme_df


def split_metabolites(expression):
    parts = expression.split(" <=> ")
    reactants = parts[0].strip().split(" + ")
    products = parts[1].strip().split(" + ")
    reactant_df = pd.DataFrame(reactants, columns=['Reactants_ID'])
    reactant_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
    reactant_df.insert(loc=2, column='Measured_value', value=reaction_ID)
    for rea in range(0, len(reactants)):
        rea_involved = reactants[rea]
        rea_screened_name = np.delete(
            np.mat(Metabolites.loc[(Metabolites[list(Metabolites)[0]] == rea_involved)]), 0, 1)
        if pd.isnull(list(rea_screened_name)).all():
            reactant_df.iloc[rea, 2] = "False"
        else:
            reactant_df.iloc[rea, 2] ="True"

    product_df = pd.DataFrame(products, columns=['Products_ID'])
    product_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
    product_df.insert(loc=2, column='Measured_value', value=reaction_ID)
    for pro in range(0, len(products)):
        pro_involved = products[pro]
        pro_screened_name = np.delete(
            np.mat(Metabolites.loc[(Metabolites[list(Metabolites)[0]] == pro_involved)]), 0, 1)
        if pd.isnull(list(pro_screened_name)).all():
            product_df.iloc[pro, 2] = "False"
        else:
            product_df.iloc[pro, 2] = "True"
    return reactant_df, product_df


def sum_enzyme(df1, df2,df3):
    true_enzyme_ids = df1[df1['Measured_value']=="True"]['Enzyme_ID']
    relevant_rows = df2[df2[list(df2)[0]].isin(true_enzyme_ids)]
    sum_enzyme_mat = np.mat(relevant_rows.sum(axis=0)).transpose()[1:,:]
    miu=np.mat(df3.iloc[0,1:]).transpose()
    ribosome=np.mat(df3.iloc[1,1:]).transpose()
    sum_enzyme_mat_normalized=sum_enzyme_mat/sum_enzyme_mat[0,0]
    miu_normalized= miu/ miu[0,0]
    ribosome_normalized=ribosome/ribosome[0,0]
    enzyme_val_1=np.multiply(sum_enzyme_mat_normalized,miu_normalized)
    enzyme_val_2=np.array(np.multiply(enzyme_val_1,ribosome_normalized),dtype=float)
    return enzyme_val_2



def r_fvalue(reaction_ID, Fluxes):
    Flux_first_value_matrix = Fluxes.loc[(Fluxes[list(Fluxes)[0]] == reaction_ID)]
    Flux_second_value_matrix = np.array(np.mat(Flux_first_value_matrix)[:,1:].transpose()).astype(float)
    return Flux_second_value_matrix


def r_mvalue(proposed_reactant_ID,km_df,Metabolites):
    met_matrix = np.mat(np.zeros(Metabolites.shape[1]-1))
    met_list=[]
    for reactant in proposed_reactant_ID:
        if not km_df.loc[km_df['Reactant_ID'] == reactant, 'Km'].iloc[0] == "nan":
            temp = np.mat(Metabolites.loc[(Metabolites[list(Metabolites)[0]]==reactant)])[:,1:]/float(km_df.loc[km_df['Reactant_ID'] == reactant, 'Km'].iloc[0])
            met_matrix = np.append(met_matrix,temp,axis=0)
            met_list=met_list+[reactant]
    met_matrix_1=np.log(np.array(np.delete(met_matrix,0,0).transpose(),dtype='float'))
    return met_matrix_1,met_list




def generate_matrix(km_df, metabolites_df, reactant_list):
    result_matrix = pd.DataFrame()

    for reactant in reactant_list:
        if not pd.isna(km_df.loc[km_df['Reactants_ID'] == reactant, 'Km'].iloc[0]):
            temp = metabolites_df[metabolites_df['Reactants_ID'] == reactant].iloc[0] / km_df.loc[km_df['Reactants_ID'] == reactant, 'Km'].iloc[0]
            result_matrix = result_matrix.append(pd.DataFrame([temp]), ignore_index=True)


def a_dataframe(reaction_ID, potential_All_reg, Metabolites):
    a_enzyme_involved = potential_All_reg.loc[(potential_All_reg['Reaction_ID'] == reaction_ID)]
    inh_involved = list(a_enzyme_involved.iloc[:, 1])
    act_involved = list(a_enzyme_involved.iloc[:, 2])

    if pd.isnull(inh_involved):
        inhibitors_df = pd.DataFrame(
            columns=['Reaction_ID', 'Inhibitors_ID', 'Inhibitors_value'])
        inhibitors_df['Inhibitors_ID'] = np.array(['False'])
        inhibitors_df['Inhibitors_value'] = np.array(['False'])
        inhibitors_df['Reaction_ID'] = reaction_ID
    else:
        inhibitors_df = pd.DataFrame(inh_involved[0].split(),
            columns=['Inhibitors_ID'])
        inhibitors_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
        inhibitors_df.insert(loc=2, column='Inhibitors_value', value=reaction_ID)
        for inh in range(len(inh_involved[0].split())):
            inh_screened_name = np.delete(
                np.mat(Metabolites.loc[(Metabolites[list(Metabolites)[0]] == inh_involved[0].split()[inh])]),
                0, 1)
            if pd.isnull(list(inh_screened_name)).all():
                inhibitors_df.iloc[inh, 2] = "False"
            else:
                inhibitors_df.iloc[inh, 2] = "True"

    if pd.isnull(act_involved):
        activitors_df = pd.DataFrame(
            columns=['Reaction_ID', 'Activators_ID', 'Activators_value'])
        activitors_df['Activators_ID'] = np.array(['False'])
        activitors_df['Activators_value'] = np.array(['False'])
        activitors_df['Reaction_ID'] = reaction_ID
    else:
        activitors_df = pd.DataFrame(act_involved[0].split(),
                                     columns=['Activators_ID'])
        activitors_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
        activitors_df.insert(loc=2, column='Activators_value', value=reaction_ID)
        for act in range(len(act_involved[0].split())):
            act_screened_name = np.delete(
                np.mat(Metabolites.loc[(Metabolites[list(Metabolites)[0]] == act_involved[0].split()[act])]),
                0, 1)
            if pd.isnull(list(act_screened_name)).all():
                activitors_df.iloc[act, 2] = "False"
            else:
                activitors_df.iloc[act, 2] = "True"
    return inhibitors_df, activitors_df


def km_dataframe(reaction_ID,Km_DF):
    selected_km = Km_DF.loc[(Km_DF[list(Km_DF)[0]]== reaction_ID)]
    reactants_value_1 = str(selected_km['Reactant_ID'].values).strip('['']')
    reactants_value_2 = (reactants_value_1.strip("''")).split(" ")
    reactants_df = pd.DataFrame(reactants_value_2, columns=['Reactant_ID'])
    km_values_1 = str(selected_km['Km'].values).strip('['']')
    km_values_2=(km_values_1.strip("''")).split(" ")
    km_values_df=pd.DataFrame(km_values_2, columns=['Km'])
    km_df = pd.concat([reactants_df, km_values_df], axis=1)
    km_df.insert(loc=0, column='Reaction_ID', value=reaction_ID)
    return km_df


def construct_total_df(allosteric_list):
    enz_sub_pro_list = ['enz_sub_pro']
    total_dataframe = pd.DataFrame(index=range(len(allosteric_list)  + len(enz_sub_pro_list)),
        columns=['Reaction_ID',  'Potential_regulator', 'Pearson_coefficient',
                 'Root_mean_square_error',
                 'P_value', 'WAIC', 'Best_regulator', 'regulator_grade',
                 'Potential_function', 'Convergence'])
    for i in total_dataframe.index:
       total_dataframe.iloc[i, 1] = (enz_sub_pro_list +  allosteric_list)[
            i % (len(allosteric_list) +  len(enz_sub_pro_list))]
    total_dataframe['Reaction_ID'] = reaction_ID
    return total_dataframe


def ln_lik_est(observed_flux, predicted_flux, RMSE):
    mle_matrix_1 = np.square((np.mat(observed_flux) - np.mat(predicted_flux)) / (1.41421 * RMSE))
    mle_matrix_2 = np.mat(np.log(RMSE))
    mle_matrix_3 = -1 / 2 * np.mat(np.full(shape=(observed_flux.shape[0], 1), fill_value=1.837877))
    mle_matrix_4 = (mle_matrix_3 - mle_matrix_2 - mle_matrix_1).sum(axis=0)
    return mle_matrix_4

def r_savalue(allosteric_reg,Metabolites):
    allosteric_reg_matrix_1=Metabolites.loc[(Metabolites[list(Metabolites)[0]] == allosteric_reg)]  # match the a allosteric regulator ID to a dataframe ("Metabolites") to acquire the measurements of the allosteric regulator
    allosteric_reg_matrix2 = np.delete(np.mat(allosteric_reg_matrix_1), 0,
                                             1)
    allosteric_reg_matrix3 = np.log(np.array(allosteric_reg_matrix2, dtype='float'))
    allosteric_reg_matrix4 = (np.array(np.transpose(allosteric_reg_matrix3))).astype(float)
    return allosteric_reg_matrix4


def r_avalue(allosteric_value_list,Metabolites):
    allosteric_first_value_matrix = np.mat(np.ones(Metabolites.shape[1]-1))       # construct a matrix to contain allosteric regulator concentrations
    for i in allosteric_value_list:
        allosteric_second_value_matrix = Metabolites.loc[(Metabolites[list(Metabolites)[0]] == i)]  # match the allosteric regulators ID to a dataframe ("Metabolites") to acquire the measurements of the allosteric regulators

        allosteric_third_value_matrix = np.delete(np.mat(allosteric_second_value_matrix), 0,
                                                  1)
        allosteric_first_value_matrix = np.append(allosteric_first_value_matrix, allosteric_third_value_matrix,
                                                  axis=0)

    allosteric_fifth_value_matrix = np.delete(allosteric_first_value_matrix, 0, 0)
    allosteric_sixth_value_matrix = np.log(np.array(allosteric_fifth_value_matrix, dtype='float'))
    allosteric_eighth_value_matrix = (np.array(np.transpose(allosteric_sixth_value_matrix))).astype(float)
    return allosteric_eighth_value_matrix

def logp(ux,lx,mu,sigma):

    cdf_up = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(ux))
    cdf_low = pm.math.exp(pm.Normal.dist(mu,sigma).logcdf(lx))
    return pm.math.log(cdf_up-cdf_low)-pm.math.log(ux-lx)




@as_op(itypes=[tt.dmatrix, tt.dvector], otypes=[tt.dvector])
def nnls_theano(mat, mu):
    return optimize.nnls(mat, mu)[0]



def all_nophos_ana(proposed_allosteric_ID,proposed_inhibitors_ID,proposed_activators_ID,reactant_list, enz_value,
                   reactant_value, flux_value,summary_df_1,subtotal_df,screen_df,screen_df_1,excel_1,excel_2,files_name):

    # prepare a series of variable for the model
    best_allo = []
    enz_ln_MAP = []
    iteration = 0

    p_value = "Nan"
    pea_coe = 'Nan'
    m_squ_err = 'Nan'

    potential_all_list = list()
    potential_inh_list = list()
    potential_act_list = list()

    # Do MCMC sample and compare the generalized model with model with regulators
    bestmodel = find_all_bestmodel(proposed_allosteric_ID, proposed_inhibitors_ID, proposed_activators_ID,reactant_list, enz_value,
                                   reactant_value, flux_value, summary_df_1, subtotal_df, screen_df,
                                   screen_df_1, excel_1, excel_2, best_allo, enz_ln_MAP, p_value, pea_coe, m_squ_err,
                                   iteration, potential_all_list, potential_inh_list, potential_act_list, files_name,
                                   lowerModel=None)

    # if bestmodel.empty:  # MCMC sample makes mistake
    #     total_matrix_1 = np.full(shape=(1, 4), fill_value='Nan')
    #     total_df = pd.DataFrame(total_matrix_1,
    #                                    columns=['potential_regulator', 'p_value', 'pearson_coefficient',
    #                                             'root_mean_square_error'])
    #     reaction_ID_matrix = np.full(shape=(1, 1), fill_value=reaction_ID)
    #     reaction_ID_df = pd.DataFrame(reaction_ID_matrix, columns=['Reaction_ID'])
    #     total_matrix_2 = pd.concat([reaction_ID_df,total_df],
    #                                axis=1)
    #     summary_df_1 = pd.concat([summary_df_1, total_matrix_2])
    #
    #
    # else:  # MCMC sample is successfully completed
    #     df_list = []
    #     for i in range(len(bestmodel.index)):
    #         df_list.append(i)
    #     total_df = bestmodel
    #     total_df.index = df_list
    #     reaction_ID_matrix = np.full(shape=(len(bestmodel.index), 1), fill_value=reaction_ID)
    #     reaction_ID_df = pd.DataFrame(reaction_ID_matrix, columns=['Reaction_ID'])
    #     total_matrix_2 = pd.concat([reaction_ID_df,  total_df],
    #                                axis=1)
    #     summary_df_1 = pd.concat([summary_df_1, total_matrix_2])
    #
    # return summary_df_1



def find_all_bestmodel(proposed_allosteric_ID, proposed_inhibitors_ID, proposed_activators_ID, reactant_list,enz_value,
                                   reactant_value, flux_value, summary_df_1, subtotal_df, screen_df,
                                   screen_df_1, excel_1, excel_2, best_results, enz_ln_MAP, p_value, pea_coe, m_squ_err,
                                   iteration, potential_all_list, potential_inh_list, potential_act_list, files_name,
                                   lowerModel=None):

    if lowerModel is None:  # this is the first round
        models, traces, loos = OrderedDict(), OrderedDict(), OrderedDict()
        compareDict, nameConvDict = dict(), dict()
        try:
            with pm.Model() as models['enz_sub_pro']:

                flux_ratio_enzyme_obs = np.array(np.log(flux_value) - np.log(enzyme_value)).astype(
                    float)

                dataframe1 = pd.DataFrame(flux_ratio_enzyme_obs, columns=["observed value"])

                # ln_j_obs_low = np.array(flux_ratio_enzyme_obs - flux_ratio_enzyme_obs * 0.1).astype(
                #     float)
                # ln_j_obs_up = np.array(flux_ratio_enzyme_obs + flux_ratio_enzyme_obs * 0.1).astype(
                #     float)

                reactants_kinetic_order = pm.Uniform("_".join(reactant_list) + "_alpha", lower=0, upper=5,
                                                     shape=(reactant_value.shape[
                                                                1], 1))  # uniform distribution for the kinetic order
                reactants_multi_kinetic_order = pm.math.dot(reactant_value, reactants_kinetic_order)
                reactants_multi_kinetic_order_1 = pm.math.exp(reactants_multi_kinetic_order)
                reactants_multi_kinetic_order_2 = tt.patternbroadcast(reactants_multi_kinetic_order_1, (False, False))

                # potential_Kcatmin = (np.array(np.log(flux_value)[0, 0])).astype(float)
                #
                # ln_kcat = pm.Uniform('ln_kcat', lower=potential_Kcatmin,
                #                      upper=2.3026 + potential_Kcatmin)  # uniform distribution for the ln_kcat
                flux_enzyme = flux_value / enzyme_value
                ln_kcat = pm.Deterministic("ln_kcat", pm.math.log(
                    nnls_theano(reactants_multi_kinetic_order_2, tt.as_tensor_variable(flux_enzyme.ravel()))))

                flux_ratio_enzyme_pre = pm.Deterministic('flux_P', ln_kcat + reactants_multi_kinetic_order)

                model_obs = pm.Normal("model_obs", mu=flux_ratio_enzyme_pre, sigma=0.1, observed=flux_ratio_enzyme_obs)

                # flux_RMSE = pm.Deterministic('flux_RMSE',
                #                              pm.math.sqrt(
                #                                  ((flux_ratio_enzyme_pre - flux_ratio_enzyme_obs) ** 2).mean()))
                #
                # flux_likelihood = pm.Normal.dist(flux_ratio_enzyme_pre, flux_RMSE)
                #
                # model_obs = pm.DensityDist("model_obs", logp,
                #                            observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low, 'mu': flux_ratio_enzyme_pre,
                #                                      'sigma': flux_RMSE},
                #                            random=flux_likelihood.random)

                traces['enz_sub_pro'] = pm.sample(5000, tune=95000, cores=2, progressbar=False)
                traceplot = pm.summary(traces['enz_sub_pro'])
                print(traceplot)
                dataframe2 = pd.DataFrame(traceplot)

                # acquire the predicted flux
                flux_ratio_enzyme_pos_dis_mat = np.mat(
                    traceplot.iloc[reactant_value.shape[1] + 1:reactant_value.shape[1] + 1 + reactant_value.shape[0],
                    0])
                dataframe3 = pd.DataFrame(np.transpose(flux_ratio_enzyme_pos_dis_mat), columns=['predicted value'])

                flux_ratio_enzyme_MSE_pos_dis_mat = np.sqrt(
                    ((flux_ratio_enzyme_obs - np.array(np.transpose(flux_ratio_enzyme_pos_dis_mat))) ** 2).sum(
                        axis=0) / (
                        np.abs(reactant_value.shape[0])))

                m_squ_err = flux_ratio_enzyme_MSE_pos_dis_mat[0]
                dataframe8 = pd.DataFrame(flux_ratio_enzyme_MSE_pos_dis_mat, columns=['root mean squared error'])

                # pm.traceplot(traces['enz_sub_pro'])
                # plt.savefig(
                #     files_name + '/parameters/' + reaction_ID + '/met_int_kcat_coe.eps', dpi=600, format='eps')
                # plt.close('all')

                flux_ratio_enzyme_obs_mat = np.squeeze(flux_ratio_enzyme_obs)
                fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                fit_model.fit(np.transpose(flux_ratio_enzyme_pos_dis_mat), flux_ratio_enzyme_obs_mat)
                fit_model_coe = np.mat(
                    fit_model.score(np.transpose(flux_ratio_enzyme_pos_dis_mat), flux_ratio_enzyme_obs_mat))
                dataframe4 = pd.DataFrame(fit_model_coe, columns=["determined coefficient"])
                dataframe5 = pd.DataFrame(fit_model.coef_, columns=['slope'])
                fit_model_predicted = fit_model.predict(np.transpose(flux_ratio_enzyme_pos_dis_mat))
                font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                plt.scatter(np.transpose(flux_ratio_enzyme_pos_dis_mat).tolist(),
                            flux_ratio_enzyme_obs_mat.tolist(), c='g', marker='o', s=40)
                plt.plot(np.transpose(flux_ratio_enzyme_pos_dis_mat.tolist()), fit_model_predicted, c='r')
                plt.yticks(fontproperties='Arial', size=15)
                plt.xticks(fontproperties='Arial', size=15)
                plt.tick_params(width=2, direction='in')
                plt.xlabel("predicted_value", font)
                plt.ylabel("observed_value", font)
                plt.savefig(
                    files_name + '/parameters/' + reaction_ID + '/met_fitting_coe.eps', dpi=600, format='eps')
                plt.close('all')

                enz_ln_MAP = np.mat(ln_lik_est(flux_ratio_enzyme_obs, np.transpose(flux_ratio_enzyme_pos_dis_mat),
                                               np.transpose(np.mat(np.ones(reactant_value.shape[0]) * 0.1))))

                print(enz_ln_MAP)
                dataframe6 = pd.DataFrame(enz_ln_MAP, columns=['enzyme_MAP'])

                pearson_coe = pearsonr(np.squeeze(flux_ratio_enzyme_obs),
                                       np.squeeze(np.transpose(np.array(flux_ratio_enzyme_pos_dis_mat))))
                pea_coe = pearson_coe[0]
                dataframe7 = pd.DataFrame(pearson_coe, index=['pearson coefficient', 'p_value'])

                writer = pd.ExcelWriter(
                    files_name + '/parameters/' + reaction_ID + '/met_int_kcat_coe.xlsx')
                dataframe1.to_excel(writer, 'obs_value')
                dataframe2.to_excel(writer, 'met_int_kcat_coe_summary')
                dataframe3.to_excel(writer, 'pre_value')
                dataframe4.to_excel(writer, 'fitting_coefficient')
                dataframe5.to_excel(writer, 'fitting_slope')
                dataframe6.to_excel(writer, 'enz_ln_MAP')
                dataframe7.to_excel(writer, 'pearson_correlation')
                dataframe8.to_excel(writer, 'root_mean_squ_err')
                writer.save()

                enz_index = subtotal_df[
                    (subtotal_df.Potential_regulator == 'enz_sub_pro')].index.tolist()
                subtotal_df.iloc[enz_index[0], 2] = pearson_coe[0]
                subtotal_df.iloc[enz_index[0], 3] = m_squ_err
                subtotal_df.iloc[enz_index[0], 4] = p_value
                if ((traceplot['r_hat'] > 1.05).sum()) > 0 or ((traceplot['r_hat'] < 0.95).sum()) > 0:
                    subtotal_df.iloc[enz_index[0], 9] = 'No'
                else:
                    subtotal_df.iloc[enz_index[0], 9] = 'Yes'



        except:
            print('something error has happened, the program will start from next reference')
            return screen_df_1

        compareDict[models['enz_sub_pro']] = traces['enz_sub_pro']
        nameConvDict[models['enz_sub_pro']] = 'enz_sub_pro'
        compRst = pm.compare(compareDict)
        best_md_loc = compRst.index[compRst['rank'] == 0][0]
        best_results.append(nameConvDict[best_md_loc])
        best_tc_loc = traces[nameConvDict[best_md_loc]]
        best_md = (best_md_loc, best_tc_loc)
        return find_all_bestmodel(proposed_allosteric_ID, proposed_inhibitors_ID, proposed_activators_ID,reactant_list, enz_value,
                                   reactant_value,flux_value, summary_df_1, subtotal_df, screen_df,
                                   screen_df_1, excel_1, excel_2, best_results, enz_ln_MAP, p_value, pea_coe, m_squ_err,
                                   iteration, potential_all_list, potential_inh_list, potential_act_list, files_name,
                                  best_md)

    else:

        assert best_results
        iteration = iteration + 1
        model, traces, loos = OrderedDict(), OrderedDict(), OrderedDict()
        compareDict, nameConvDict, p_valueDict, correl_Dict, error_Dict, ln_MAP_Dict = dict(), dict(), dict(), dict(), dict(), dict()
        candidate_all_list = []

        for potential_all_reg in proposed_allosteric_ID:  # judge allosteric regulators one by one
            print(potential_all_reg)
            try:
                with pm.Model() as model[potential_all_reg]:
                    flux_ratio_enzyme_obs = np.array(np.log(flux_value) - np.log(enzyme_value)).astype(float)
                    # ln_j_obs_low = np.array(flux_ratio_enzyme_obs - flux_ratio_enzyme_obs * 0.1).astype(
                    #     float)
                    # ln_j_obs_up = np.array(flux_ratio_enzyme_obs + flux_ratio_enzyme_obs * 0.1).astype(
                    #     float)

                    dataframe1 = pd.DataFrame(flux_ratio_enzyme_obs, columns=["observed value"])

                    reactants_kinetic_order = pm.Uniform(
                        "_".join(reactant_list) + "_" + potential_all_reg + "_alpha", lower=0, upper=5,
                        shape=(reactant_value.shape[1], 1))  # uniform distribution for the kinetic order
                    reactants_mul_coe = pm.math.dot(reactant_value, reactants_kinetic_order)

                    additional_parameter = 0  # a variable that indicates how many parameters are added
                    current_potential_all_reg_value = r_savalue(potential_all_reg,
                                                                Metabolites)  # determine the value for potential allosteric regulators

                    current_potential_all_reg_value_median = np.median(current_potential_all_reg_value, axis=0)

                    current_log_Km_value = pm.Uniform(potential_all_reg + '_c_log_Km',
                                                      lower=-15 + current_potential_all_reg_value_median,
                                                      upper=15 + current_potential_all_reg_value_median,
                                                      shape=(current_potential_all_reg_value.shape[1], 1))

                    # judge the function of the potential allosteric regulators according to the prior knowledge
                    if (potential_all_reg in proposed_inhibitors_ID and potential_all_reg in proposed_activators_ID):
                        current_kinetic_order = pm.Uniform(potential_all_reg + '_c_alpha', lower=-5,
                                                           upper=5, shape=(current_potential_all_reg_value.shape[1], 1))


                    elif potential_all_reg in proposed_inhibitors_ID and potential_all_reg not in proposed_activators_ID:
                        current_kinetic_order = pm.Uniform(potential_all_reg + '_c_alpha', lower=-5,
                                                           upper=0,
                                                           shape=(current_potential_all_reg_value.shape[1], 1))

                    elif potential_all_reg not in proposed_inhibitors_ID and potential_all_reg in proposed_activators_ID:
                        current_kinetic_order = pm.Uniform(potential_all_reg + '_c_alpha', lower=0,
                                                           upper=5,
                                                           shape=(current_potential_all_reg_value.shape[1], 1))
                    cur_reg_km = tt.as_tensor_variable(current_potential_all_reg_value) - current_log_Km_value

                    current_all_mul_coe = pm.math.dot(cur_reg_km, current_kinetic_order)

                    additional_parameter = additional_parameter + 2

                    # judge whether the plausible allosteric regulators exist after multiple rounds
                    if len(potential_all_list) == 0:
                        potential_all_mul_coe = (np.array(np.zeros(reactant_value.shape[0]))).astype(float)
                        potential_all_mul_coe = tt.zeros((13, 1), dtype='float64')
                        print(potential_all_mul_coe)
                        additional_parameter = additional_parameter
                    else:
                        potential_allosteric_value_matrix = r_avalue(potential_all_list, Metabolites)
                        potential_allosteric_value_matrix_median = np.median(potential_allosteric_value_matrix, axis=0)
                        allsoteric_log_Km_value = pm.Uniform(potential_all_reg + '_al_log_Km',
                                                             lower=np.array(np.mat(
                                                                 -15 + potential_allosteric_value_matrix_median)).astype(
                                                                 float),
                                                             upper=np.array(np.mat(
                                                                 15 + potential_allosteric_value_matrix_median)).astype(
                                                                 float),
                                                             shape=(1, len(
                                                                 potential_all_list)))  # uniform distribution for Km

                        allosteric_kinetic_order = pm.Uniform(potential_all_reg + '_al_alpha',
                                                              lower=-5, upper=5, shape=(len(
                                potential_all_list), 1))  # uniform distribution for kinetic order

                        all_reg_km = potential_allosteric_value_matrix - allsoteric_log_Km_value
                        potential_all_mul_coe = pm.math.dot(all_reg_km, allosteric_kinetic_order)
                        additional_parameter = additional_parameter + len(potential_all_list) * 2

                    # judge whether the plausible inhibitors exist after multiple rounds
                    if len(potential_inh_list) == 0:
                        potential_inh_mul_coe = (np.array(np.zeros(reactant_value.shape[0]))).astype(float)
                        potential_inh_mul_coe = tt.zeros((13, 1), dtype='float64')
                        print(potential_inh_mul_coe)
                        additional_parameter = additional_parameter
                    else:
                        potential_inhibitors_value_matrix = r_avalue(potential_inh_list, Metabolites)
                        print("inhibitors:", potential_inhibitors_value_matrix)

                        potential_inhibitors_value_matrix_median = np.median(potential_inhibitors_value_matrix, axis=0)
                        print("median:", potential_inhibitors_value_matrix_median)
                        inhibitors_log_Km_value = pm.Uniform(potential_all_reg + '_i_log_Km',
                                                             lower=np.array(np.mat(
                                                                 -15 + potential_inhibitors_value_matrix_median)).astype(
                                                                 float),
                                                             upper=np.array(np.mat(
                                                                 15 + potential_inhibitors_value_matrix_median)).astype(
                                                                 float),
                                                             shape=(1, len(
                                                                 potential_inh_list)))  # uniform distribution for Km
                        inhibitors_kinetic_order = pm.Uniform(potential_all_reg + '_i_alpha',
                                                              lower=-5, upper=0, shape=(len(
                                potential_inh_list), 1))  # uniform distribution for kinetic order
                        inh_reg_km = potential_inhibitors_value_matrix - inhibitors_log_Km_value

                        potential_inh_mul_coe = pm.math.dot(inh_reg_km, inhibitors_kinetic_order)
                        additional_parameter = additional_parameter + len(potential_inh_list) * 2

                    # judge whether the plausible activators exist after multiple rounds
                    if len(potential_act_list) == 0:
                        potential_act_mul_coe = (np.array(np.zeros(reactant_value.shape[0]))).astype(float)
                        potential_act_mul_coe = tt.zeros((13, 1), dtype='float64')
                        print(potential_act_mul_coe)
                        additional_parameter = additional_parameter
                    else:
                        potential_activators_value_matrix = r_avalue(potential_act_list, Metabolites)

                        potential_activators_value_matrix_median = np.median(potential_activators_value_matrix, axis=0)
                        activators_log_Km_value = pm.Uniform(potential_all_reg + '_ac_log_Km',
                                                             lower=np.array(np.mat(
                                                                 -15 + potential_activators_value_matrix_median)).astype(
                                                                 float),
                                                             upper=np.array(np.mat(
                                                                 15 + potential_activators_value_matrix_median)).astype(
                                                                 float),
                                                             shape=(1, len(
                                                                 potential_act_list)))  # uniform distribution for Km
                        activators_kinetic_order = pm.Uniform(potential_all_reg + '_ac_alpha',
                                                              lower=0, upper=5, shape=(len(
                                potential_act_list), 1))  # uniform distribution for kinetic order

                        act_reg_km = potential_activators_value_matrix - activators_log_Km_value
                        potential_act_mul_coe = pm.math.dot(act_reg_km, activators_kinetic_order)
                        additional_parameter = additional_parameter + len(potential_act_list) * 2

                    print(current_potential_all_reg_value)

                    all_mul_coe = reactants_mul_coe + current_all_mul_coe + tt.as_tensor_variable(
                        potential_all_mul_coe) + tt.as_tensor_variable(potential_inh_mul_coe) + tt.as_tensor_variable(
                        potential_act_mul_coe)

                    all_mul_coe_1 = pm.math.exp(all_mul_coe)
                    all_mul_coe_2 = tt.patternbroadcast(all_mul_coe_1, (False, False))
                    flux_enzyme = flux_value / enzyme_value
                    ln_kcat = pm.Deterministic(potential_all_reg + '_ln_kcat', pm.math.log(
                        nnls_theano(all_mul_coe_2, tt.as_tensor_variable(flux_enzyme.ravel()))))

                    # potential_Kcatmin = (np.array(np.log(flux_value)[0, 0])).astype(float)
                    # ln_kcat = pm.Uniform(potential_all_reg + '_ln_kcat', lower=potential_Kcatmin,
                    #                      upper=2.3026 + potential_Kcatmin)  # uniform distribution for the ln_kcat

                    # second: Calculate predicted flux
                    flux_ratio_enzyme_pre = pm.Deterministic(potential_all_reg + '_flux',
                                                             ln_kcat + all_mul_coe)

                    model_obs = pm.Normal("model_obs", mu=flux_ratio_enzyme_pre, sigma=0.1,
                                          observed=flux_ratio_enzyme_obs)

                    # flux_RMSE = pm.Deterministic(potential_all_reg + '_RMSE',
                    #                              pm.math.sqrt(
                    #                                  ((flux_ratio_enzyme_pre - flux_ratio_enzyme_obs) ** 2).sum(
                    #                                      axis=0) / pm.math.abs_(reactant_value.shape[0])))
                    #
                    # # third: Calculate the likelihood of kinetic parameter
                    # flux_likelihood = pm.Normal.dist(flux_ratio_enzyme_pre, flux_RMSE)

                    # model_obs = pm.DensityDist("model_obs", logp,
                    #                            observed={'ux': ln_j_obs_up, 'lx': ln_j_obs_low,
                    #                                      'mu': flux_ratio_enzyme_pre,
                    #                                      'sigma': flux_RMSE},
                    #                            random=flux_likelihood.random)

                    # fourth: Evaluate the posterior probability of the kinetic parameter to determine whether the kinetic parameter we are drawn should be accepted or rejected
                    # start = {'as_1360_c_alpha': 0.5, 'as_1360_c_log_Km': -1, 's_0434_s_0557_as_1360_alpha': 1}

                    traces[potential_all_reg] = pm.sample(5000, tune=95000,
                                                          cores=2,
                                                          progressbar=False)  # Here we used NUTS algorithm as the model we use is nonlinear kinetic equation

                    traceplot = pm.summary(traces[potential_all_reg])
                    print(traceplot)
                    dataframe2 = pd.DataFrame(traceplot)

                    # acquire the predicted flux
                    flux_ratio_enzyme_pos_dis_mat = np.mat(
                        traceplot.iloc[
                        len(reactant_list) + additional_parameter + 1: len(reactant_list) + additional_parameter + 1 +
                                                                       reactant_value.shape[0], 0])
                    print(flux_ratio_enzyme_pos_dis_mat)

                    dataframe3 = pd.DataFrame(np.transpose(flux_ratio_enzyme_pos_dis_mat), columns=['predicted value'])
                    flux_ratio_enzyme_MSE_pos_dis_mat = np.sqrt(
                        ((flux_ratio_enzyme_obs - np.array(np.transpose(flux_ratio_enzyme_pos_dis_mat))) ** 2).sum(
                            axis=0) / (
                            np.abs(reactant_value.shape[0])))

                    # plot a posteriori distribution figure
                    # pm.traceplot(traces[potential_all_reg])
                    # plt.savefig(
                    #     files_name + '/parameters/' + reaction_ID + '/' + "_".join(
                    #         best_results) + '-' + potential_all_reg + '-' + 'met_int_kcat_coe.eps', dpi=600, format='eps')
                    # plt.close('all')

                    flux_ratio_enzyme_obs_mat = np.squeeze(flux_ratio_enzyme_obs)
                    fit_model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
                    fit_model.fit(np.transpose(flux_ratio_enzyme_pos_dis_mat), flux_ratio_enzyme_obs_mat)
                    fit_model_coe = np.mat(
                        fit_model.score(np.transpose(flux_ratio_enzyme_pos_dis_mat), flux_ratio_enzyme_obs_mat))
                    dataframe4 = pd.DataFrame(fit_model_coe, columns=["determined coefficient"])
                    dataframe5 = pd.DataFrame(fit_model.coef_, columns=['slope'])
                    fit_model_predicted = fit_model.predict(np.transpose(flux_ratio_enzyme_pos_dis_mat))
                    font = {'family': 'Arial', 'weight': 'normal', 'size': 15, }
                    plt.scatter(np.transpose(flux_ratio_enzyme_pos_dis_mat).tolist(),
                                flux_ratio_enzyme_obs_mat.tolist(), c='g', marker='o', s=40)
                    plt.plot(np.transpose(flux_ratio_enzyme_pos_dis_mat.tolist()), fit_model_predicted, c='r')
                    plt.yticks(fontproperties='Arial', size=15)
                    plt.xticks(fontproperties='Arial', size=15)
                    plt.tick_params(width=2, direction='in')
                    plt.xlabel("predicted_value", font)
                    plt.ylabel("observed_value", font)
                    plt.savefig(
                        files_name + '/parameters/' + reaction_ID + '/' + "_".join(
                            best_results) + '-' + potential_all_reg + '-' + 'met_fitting_coe.eps', dpi=600,
                        format='eps')
                    plt.close('all')

                    # calculate maximum a posterior of the model
                    all_ln_MAP = np.mat(ln_lik_est(flux_ratio_enzyme_obs, np.transpose(flux_ratio_enzyme_pos_dis_mat),
                                                   np.transpose(np.mat(np.ones(reactant_value.shape[0]) * 0.1))))

                    print(all_ln_MAP)
                    dataframe6 = pd.DataFrame(all_ln_MAP, columns=['allosteric_MAP'])

                    # Do pearson correlation between predicted flux and observed flux
                    pearson_coe = pearsonr(np.squeeze(flux_ratio_enzyme_obs),
                                           np.squeeze(np.transpose(np.array(flux_ratio_enzyme_pos_dis_mat))))
                    dataframe7 = pd.DataFrame(pearson_coe, index=['pearson coefficient', 'p_value'])

                    # calculate the root mean squared error (RMSE) between predicted flux and observed flux
                    root_mean_squ_err = np.sqrt(
                        ((flux_ratio_enzyme_obs - np.transpose(np.array(flux_ratio_enzyme_pos_dis_mat))) ** 2).sum(
                            axis=0) /
                        reactant_value.shape[0])
                    dataframe8 = pd.DataFrame(root_mean_squ_err, columns=['root mean squared error'])

                    # do likelihood ratio test
                    all_p_value = chi2.sf(2 * (all_ln_MAP - enz_ln_MAP), 2)
                    print(all_p_value)
                    dataframe9 = pd.DataFrame(all_p_value, columns=['p_value'])

                    # save the results of the Bayesian inference
                    writer = pd.ExcelWriter(
                        files_name + '/parameters/' + reaction_ID + '/' + "_".join(
                            best_results) + '-' + potential_all_reg + '-' + 'met_int_kcat_coe.xlsx')
                    dataframe1.to_excel(writer, 'flux_obs')
                    dataframe2.to_excel(writer, 'met_int_kcat_coe_summary')
                    dataframe3.to_excel(writer, 'flux_pos_dis_data')
                    dataframe4.to_excel(writer, 'fitting_coefficient')
                    dataframe5.to_excel(writer, 'fitting_slope')
                    dataframe6.to_excel(writer, 'all_ln_MAP')
                    dataframe7.to_excel(writer, 'pearson_correlation')
                    dataframe8.to_excel(writer, 'root_mean_squ_err')
                    dataframe9.to_excel(writer, 'p_value')
                    writer.save()

                    # Summarize the results into the given data frame (subtotal_dataframe)
                    if iteration >= 2:  # multiple results are not summarized
                        pass
                    else:
                        enz_index = subtotal_df[
                            (subtotal_df.Potential_regulator == potential_all_reg)].index.tolist()
                        subtotal_df.iloc[enz_index[0], 2] = pearson_coe[0]
                        subtotal_df.iloc[enz_index[0], 3] = root_mean_squ_err[0]
                        subtotal_df.iloc[enz_index[0], 4] = all_p_value[0, 0]
                        if (
                                potential_all_reg in proposed_inhibitors_ID and potential_all_reg in proposed_activators_ID):
                            subtotal_df.iloc[enz_index[0], 8] = 'Inhibition or Activation'

                        elif potential_all_reg in proposed_inhibitors_ID and potential_all_reg not in proposed_activators_ID:
                            subtotal_df.iloc[enz_index[0], 8] = 'Inhibition'
                        elif potential_all_reg not in proposed_inhibitors_ID and potential_all_reg in proposed_activators_ID:
                            subtotal_df.iloc[enz_index[0], 8] = 'Activation'

                        if ((traceplot['r_hat'] > 1.05).sum()) > 0 or ((traceplot['r_hat'] < 0.95).sum()) > 0:
                            subtotal_df.iloc[enz_index[0], 9] = 'No'
                        else:
                            subtotal_df.iloc[enz_index[0], 9] = 'Yes'





            except:
                print('something error has happened, next potential regulators will be evaluated')
                continue

            # screen candidate regulators from a series of potential regulators

            if all_p_value[0, 0] < 0.1:
                compareDict[model[potential_all_reg]] = traces[potential_all_reg]
                nameConvDict[model[potential_all_reg]] = potential_all_reg
                p_valueDict[model[potential_all_reg]] = all_p_value[0, 0]
                correl_Dict[model[potential_all_reg]] = pearson_coe[0]
                error_Dict[model[potential_all_reg]] = root_mean_squ_err[0]
                ln_MAP_Dict[model[potential_all_reg]] = all_ln_MAP[0, 0]
                candidate_all_list.append(potential_all_reg)

        compareDict[lowerModel[0]] = lowerModel[1]
        nameConvDict_df = pd.DataFrame(nameConvDict, index=[0])
        p_valueDict_df = pd.DataFrame(p_valueDict, index=[0])
        correl_Dict_df = pd.DataFrame(correl_Dict, index=[0])
        error_Dict_df = pd.DataFrame(error_Dict, index=[0])
        ln_MAP_Dict_df = pd.DataFrame(ln_MAP_Dict, index=[0])

        # construct a data frame to include the results of model comparision
        connect_df_1 = pd.concat(
            [nameConvDict_df, p_valueDict_df, correl_Dict_df, error_Dict_df,
             ln_MAP_Dict_df],
            axis=0)
        connect_df_1.index = ['potential_regulator', 'p_value', 'pearson_coefficient',
                                     'root_mean_square_error', 'MAP']
        connect_df_2 = pd.DataFrame(connect_df_1.values.T, index=connect_df_1.columns,
                                           columns=connect_df_1.index)
        best_results_series = pd.Series(
            {"potential_regulator": "_".join(best_results), "p_value": p_value, 'pearson_coefficient': pea_coe,
             'root_mean_square_error': m_squ_err, 'MAP': enz_ln_MAP[0, 0]}, name=lowerModel[0])

        connect_df_3 = connect_df_2.append(best_results_series)


        assert compareDict
        compRst_1 = pm.compare(compareDict)  # DO comparison among all the candidate regulators
        compRst_2 = pd.concat([compRst_1, connect_df_3], axis=1)
        print(compRst_2)
        connect_df_3.to_excel(excel_1, "-".join(best_results) + "-" + 'name')
        compRst_2.to_excel(excel_1, "-".join(best_results) + "-" + 'com')

        if iteration >= 2:  # multiple results are not summarized
            pass
        else:
            number = -1
            for p_reg in compRst_2["potential_regulator"].tolist():
                number = number + 1
                enz_index_2 = subtotal_df[
                    (subtotal_df.Potential_regulator == p_reg)].index.tolist()
                subtotal_df.iloc[enz_index_2[0], 5] = compRst_2['loo'][compRst_2['rank'] == number].tolist()[
                    0]

        best_md_loc = compRst_1.index[compRst_1['rank'] == 0][0]
        screen_series = compRst_2.iloc[:, 9:13]
        screen_df = (pd.concat([screen_df, screen_series], axis=0)).astype(str)

        if best_md_loc == lowerModel[0]:  # there are no regulators that best supported the reaction
            excel_1.save()
            print('Finally, found the best model is\033[1;31;43m', best_results, '\033[0m')
            subtotal_list = list(subtotal_df['Potential_regulator'])[:]
            grade = 0
            for i in best_results:
                enz_index_1 = subtotal_df[(subtotal_df.Potential_regulator == i)].index.tolist()
                subtotal_df.iloc[enz_index_1[0], 6] = "True"
                if i == "enz_sub_pro":
                    pass
                else:
                    grade = grade + 1
                    subtotal_df.iloc[enz_index_1[0], 7] = grade
                subtotal_list.remove(i)

            for j in subtotal_list:
                enz_index_2 = subtotal_df[
                    (subtotal_df.Potential_regulator == j)].index.tolist()
                subtotal_df.iloc[enz_index_2[0], 6] = "False"
            screen_df_2 = screen_df.astype(str)
            for k in range(1, len(best_results) + 1):
                temporary_df_1 = \
                    screen_df_2[screen_df_2['potential_regulator'] == "_".join(best_results[0:k])].iloc[0]
                temporary_df_2 = pd.DataFrame(pd.DataFrame(temporary_df_1).values.T,
                                                    index=pd.DataFrame(temporary_df_1).columns,
                                                    columns=pd.DataFrame(temporary_df_1).index)
                screen_df_1 = pd.concat([screen_df_1, temporary_df_2], axis=0)
            print(screen_df_1)
            subtotal_df.to_excel(excel_2, sheet_name="results_subtotal")
            excel_2.save()
            return screen_df_1

        else:  # find multiple regulators
            p_value = compRst_2.iloc[0].at['p_value']
            pea_coe = compRst_2.iloc[0].at['pearson_coefficient']
            m_squ_err = compRst_2.iloc[0].at['root_mean_square_error']
            enz_ln_MAP = np.mat(compRst_2.iloc[0].at['MAP'])
            best_tc_loc = traces[nameConvDict[best_md_loc]]
            best_md = (best_md_loc, best_tc_loc)
            best_results.append(nameConvDict[best_md_loc])
            potential_all_reg_list_1 = candidate_all_list
            potential_all_reg_list_1.remove(nameConvDict[best_md_loc])
            print("ccccccccccccccccccccccccccccccccccccccccccc")
            print(potential_all_reg_list_1)
            if nameConvDict[best_md_loc][0]=="i":
                potential_all_reg_list_2 = [item for item in potential_all_reg_list_1 if
                                            item != "a" + nameConvDict[best_md_loc][1:]]
            else:
                potential_all_reg_list_2 = [item for item in potential_all_reg_list_1 if
                                            item != "i" + nameConvDict[best_md_loc][1:]]
            print(potential_all_reg_list_2)

            if nameConvDict[best_md_loc] in proposed_inhibitors_ID and nameConvDict[
                best_md_loc] in proposed_activators_ID:
                potential_all_list = potential_all_list + nameConvDict[best_md_loc].split(' ')

            elif nameConvDict[best_md_loc] in proposed_inhibitors_ID and nameConvDict[
                best_md_loc] not in proposed_activators_ID:
                potential_inh_list = potential_inh_list + nameConvDict[best_md_loc].split(' ')

            elif nameConvDict[best_md_loc] in proposed_activators_ID and nameConvDict[
                best_md_loc] not in proposed_inhibitors_ID:
                potential_act_list = potential_act_list + nameConvDict[best_md_loc].split(' ')

            return find_all_bestmodel(potential_all_reg_list_2, proposed_inhibitors_ID, proposed_activators_ID,reactant_list, enz_value,
                                       reactant_value,  flux_value, summary_df_1, subtotal_df,
                                      screen_df,
                                      screen_df_1, excel_1, excel_2, best_results, enz_ln_MAP, p_value, pea_coe,
                                      m_squ_err,
                                      iteration, potential_all_list, potential_inh_list, potential_act_list, files_name,
                                      best_md)



def Nan_df_cons():
    """
    Create an empty data frame because the metabolites, regulators, enzymes, and reaction fluxes involved in the
    reaction have no observed values under the given conditions.
    :return: a summary data frame whose element is defined as "Nan"
    """
    summary_df_1 = pd.DataFrame(
        columns=['reaction_ID',  'potential_regulator', 'p_value', 'pearson_coefficient',
                 'root_mean_square_error'])
    total_matrix_1 = np.full(shape=(1, 4), fill_value='Nan')
    total_df = pd.DataFrame(total_matrix_1,
                                   columns=['potential_regulator', 'p_value', 'pearson_coefficient',
                                            'root_mean_square_error'])
    reaction_ID_matrix = np.full(shape=(1, 1), fill_value=reaction_ID)
    reaction_ID_df = pd.DataFrame(reaction_ID_matrix, columns=['reaction_ID'])
    total_matrix_2 = pd.concat([reaction_ID_df,  total_df], axis=1)
    summary_df_1 = pd.concat([summary_df_1, total_matrix_2])
    return summary_df_1




excel_name="ms3model.xlsx"

Reactions = pd.read_excel(excel_name,sheet_name='Reaction')
Metabolites = pd.read_excel(excel_name,sheet_name='metabolite_mean')
Transcripts = pd.read_excel(excel_name,sheet_name='transcription')
miu_Ribosome= pd.read_excel(excel_name,sheet_name='miu_ribosome')
Fluxes = pd.read_excel(excel_name, sheet_name='flux_mean')
potential_All_reg = pd.read_excel(excel_name,sheet_name='allosteric_regulators')
Km_DF = pd.read_excel(excel_name,sheet_name='Km')


summary_df = pd.DataFrame(
    columns=['reaction_ID',  'potential_regulator', 'p_value', 'pearson_coefficient', 'root_mean_square_error'])

# 示例使用
for step in range(23, len(Reactions)):
    reaction_ID = Reactions.iloc[step, 0]
    print(reaction_ID)
    enzyme_ID_df=process_enzyme_id_to_df(Reactions.iloc[step, 3])
    reactant_df, product_df=split_metabolites(Reactions.iloc[step, 2])
    km_df=km_dataframe(reaction_ID,Km_DF)
    if "True" in list(reactant_df['Measured_value']):
        proposed_enzyme_ID_1 = set(list(enzyme_ID_df['Enzyme_ID'][enzyme_ID_df['Measured_value'] == 'True']))
        if len(proposed_enzyme_ID_1) != 0:
            enzyme_value=sum_enzyme(enzyme_ID_df, Transcripts,miu_Ribosome)
            flux_value = r_fvalue(reaction_ID, Fluxes)

            if np.isnan(flux_value).all() or not np.any(flux_value):
                print(reaction_ID, "has no measured flux, next reaction will start")
                ch = Nan_df_cons()
                continue
            else:
                inhibitors_df, activitors_df = a_dataframe(reaction_ID,potential_All_reg,Metabolites)
                proposed_inhibitors_ID = list(
                    inhibitors_df['Inhibitors_ID'][inhibitors_df['Inhibitors_value'] == 'True'])
                proposed_activators_ID = list(
                    activitors_df['Activators_ID'][activitors_df['Activators_value'] == 'True'])
                proposed_product_ID = list(
                    product_df['Products_ID'][product_df['Measured_value'] == 'True'])
                if len(proposed_product_ID)==0:
                    proposed_product_ID_1=proposed_product_ID
                else:
                    proposed_product_ID_1=['i' + str(item) for item in proposed_product_ID]

                proposed_inhibitors_ID_1 = list(sorted(
                    set(proposed_inhibitors_ID + proposed_product_ID_1)))

                proposed_allosteric_ID = list(sorted(
                    set(proposed_inhibitors_ID_1+ proposed_activators_ID)))

                proposed_reactant_ID = list(
                    reactant_df['Reactants_ID'][reactant_df['Measured_value'] == 'True'])

                reactant_value, reactant_list= r_mvalue(proposed_reactant_ID,km_df,
                                         Metabolites)  # acquire the metabolite concentration under all the experimental groups



                if  len(proposed_allosteric_ID) == 0:      # all the regulators we provided are not measured under all the experimental groups

                    print(
                        'allosteric regulators have not been used in intrinsic kcat model because they are all nan in the model')
                    ch = Nan_df_cons()
                    continue

                else:
                    summary_df_1 = pd.DataFrame(
                        columns=['reaction_ID', 'potential_regulator', 'p_value', 'pearson_coefficient',
                                 'root_mean_square_error'])

                    files_name = 'MS3/allos_reg'
                    if os.path.isfile(files_name + '/statistics/' + reaction_ID + '/results subtotal.xlsx'):
                        continue
                    else:
                        if os.path.exists(
                                files_name + '/statistics/' + reaction_ID):
                            excel_2 = pd.ExcelWriter(
                                files_name + '/statistics/' + reaction_ID + '/results subtotal.xlsx')
                            subtotal_df = construct_total_df(proposed_allosteric_ID)
                        else:
                            os.makedirs(files_name + '/statistics/' + reaction_ID)
                            excel_2 = pd.ExcelWriter(
                                files_name + '/statistics/' + reaction_ID + '/results subtotal.xlsx')
                            subtotal_df = construct_total_df(proposed_allosteric_ID)

                    if os.path.isfile(
                            files_name + '/statistics/' + reaction_ID + '/results statistics.xlsx'):
                        continue
                    else:
                        if os.path.exists(files_name + '/parameters/' + reaction_ID):
                            os.rmdir(files_name + '/statistics/' + reaction_ID)
                            shutil.rmtree(
                                files_name + '/parameters/' + reaction_ID)
                        else:
                            os.rmdir(files_name + '/statistics/' + reaction_ID)

                        os.makedirs(files_name + '/parameters/' + reaction_ID)
                        os.makedirs(files_name + '/statistics/' + reaction_ID)

                        excel_1 = pd.ExcelWriter(
                            files_name + '/statistics/' + reaction_ID + '/results statistics.xlsx')

                        screen_df = pd.DataFrame(
                            columns=['potential_regulator', 'p_value', 'pearson_coefficient', 'root_mean_square_error'])
                        screen_df_1 = pd.DataFrame(
                            columns=['potential_regulator', 'p_value', 'pearson_coefficient', 'root_mean_square_error'])

                        ch = all_nophos_ana(proposed_allosteric_ID, proposed_inhibitors_ID, proposed_activators_ID,reactant_list,
                                            enzyme_value,reactant_value, flux_value,summary_df_1,subtotal_df,screen_df,screen_df_1,excel_1,excel_2,files_name)

        else:
            print(reaction_ID, "has no detective enzyme, next reaction will start")
            ch = Nan_df_cons()
            continue

    else:
        print(reaction_ID, "has no detective metabolite, next reaction will start")
        ch = Nan_df_cons()
        continue

    summary_df = pd.concat([summary_df, ch])










