"""
multiple_experiments.py
    given a list of config file run multitple experiment
"""
import subprocess
import time

energy_in = 0
energy_gamma = 0
energy_eta = 1
energy_both = 0
energy_both_gammaneg = 0
energy_both_dis_k = 0


if energy_in:
    energy_insensitive_config = ['energy_independent_seed_1',
                                'energy_independent_seed_2',
                                'energy_independent_seed_3',
                                ]
else:
    energy_insensitive_config = []

if energy_gamma:
    energy_gamma_sensitive_config = [#'energy_dependent_gamma_seed_1',
                                    'energy_dependent_gamma_seed_2',
                                    'energy_dependent_gamma_seed_3',
                                    #'energy_dependent_gamma_seed_4',
                                    #'energy_dependent_gamma_seed_5',
                                    #'energy_dependent_gamma_seed_6',
                                    #'energy_dependent_gamma_seed_7',
                                    #'energy_dependent_gamma_seed_8',
                                    #'energy_dependent_gamma_seed_9',
                                    #'energy_dependent_gamma_seed_10',
                                     ]
else:
    energy_gamma_sensitive_config = []

if energy_eta:
    energy_eta_sensitive_config = [
                                #'energy_dependent_eta_0_seed_1_n_500',
                                #'energy_dependent_eta_1_seed_1_n_500',
                                #'energy_dependent_eta_2_seed_1_n_500',
                                #'energy_dependent_eta_5_seed_1_n_500',
                                #'energy_dependent_eta_7_seed_1_n_500',
                                #'energy_dependent_eta_10_seed_1_n_500',
                                #'energy_dependent_eta_13_seed_1_n_500',
                                #'energy_dependent_eta_15_seed_1_n_500',
                                #'energy_dependent_eta_20_seed_1_n_500',
                                #'energy_dependent_eta_30_seed_1_n_500',
                                #'energy_dependent_eta_30_seed_2_n_500',
                                #'energy_dependent_eta_30_seed_3_n_500',
                                #'energy_dependent_eta_30_seed_4_n_500',
                                #'energy_dependent_eta_30_seed_5_n_500',
                                #'energy_dependent_eta_40_seed_1_n_500',
                                #'energy_dependent_eta_50_seed_1_n_500',
                                #'energy_dependent_eta_50_seed_2_n_500',
                                #'energy_dependent_eta_50_seed_3_n_500',
                                #'energy_dependent_eta_50_seed_4_n_500',
                                #'energy_dependent_eta_50_seed_5_n_500',
                                #'energy_dependent_eta_60_seed_1_n_500',
                                #'energy_dependent_eta_70_seed_1_n_500',
                                #'energy_dependent_eta_100_seed_1_n_500',
                                #'energy_dependent_eta_100_seed_2_n_500',
                                #'energy_dependent_eta_100_seed_3_n_500',
                                #'energy_dependent_eta_100_seed_4_n_500',
                                #'energy_dependent_eta_100_seed_5_n_500',
                                #'energy_dependent_eta_1_seed_2_n_500',
                                #'energy_dependent_eta_2_seed_2_n_500',
                                #'energy_dependent_eta_5_seed_2_n_500',
                                #'energy_dependent_eta_7_seed_2_n_500',
                                #'energy_dependent_eta_10_seed_2_n_500',
                                #'energy_dependent_eta_13_seed_1_n_500',
                                #'energy_dependent_eta_15_seed_1_n_500',
                                #'energy_dependent_eta_20_seed_2_n_500',
                                #'energy_dependent_eta_30_seed_2_n_500',
                                #'energy_dependent_eta_50_seed_2_n_500',
                                #'energy_dependent_eta_100_seed_2_n_500',
                                #'energy_dependent_eta_10_seed_1_n_1000',
                                #'energy_dependent_eta_seed_5',
                                "energy_dependent_eta_50_w_uniform_5_seed_1_n_500",
                                "energy_dependent_alpha_hetero_seed_1_n_500"
                                ]
else:
    energy_eta_sensitive_config = []

if energy_both:
    energy_gamma_and_eta_sensitive_config = [
                                    #'energy_dependent_both_gamma_1_seed_1_n_500',
                                    #'energy_dependent_both_gamma_2_seed_1_n_500',
                                    #'energy_dependent_both_gamma_10_seed_1_n_500',
                                    #'energy_dependent_both_gamma_20_seed_1_n_500',
                                    #'energy_dependent_both_gamma_30_seed_1_n_500',
                                    #'energy_dependent_both_gamma_50_seed_1_n_500',
                                    #'energy_dependent_both_gamma_60_seed_1_n_500',
                                    #'energy_dependent_both_gamma_100_seed_1_n_500',
                                #'energy_dependent_gamma_10_eta_30_seed_1_n_500',
                                #'energy_dependent_gamma_10_eta_40_seed_1_n_500',
                                #'energy_dependent_gamma_10_eta_50_seed_1_n_500',
                                'energy_dependent_gamma_10_eta_50_seed_2_n_500',
                                'energy_dependent_gamma_10_eta_50_seed_3_n_500',
                                'energy_dependent_gamma_10_eta_50_seed_4_n_500',
                                'energy_dependent_gamma_10_eta_50_seed_5_n_500',
                                #'energy_dependent_gamma_10_eta_60_seed_1_n_500',
                                #'energy_dependent_gamma_10_eta_70_seed_1_n_500',
                                #'energy_dependent_gamma_20_eta_30_seed_1_n_500',
                                #'energy_dependent_gamma_20_eta_40_seed_1_n_500',
                                #'energy_dependent_gamma_20_eta_50_seed_1_n_500',
                                'energy_dependent_gamma_20_eta_50_seed_2_n_500',
                                'energy_dependent_gamma_20_eta_50_seed_3_n_500',
                                'energy_dependent_gamma_20_eta_50_seed_4_n_500',
                                'energy_dependent_gamma_20_eta_50_seed_5_n_500',
                                'energy_dependent_gamma_20_eta_50_eap_10_seed_1_n_500',
                                'energy_dependent_gamma_20_eta_50_eap_20_seed_1_n_500',
                                #'energy_dependent_gamma_20_eta_60_seed_1_n_500',
                                #'energy_dependent_gamma_20_eta_70_seed_1_n_500',
                                #'energy_dependent_gamma_50_eta_30_seed_1_n_500',
                                #'energy_dependent_gamma_50_eta_40_seed_1_n_500',
                                #'energy_dependent_gamma_50_eta_50_seed_1_n_500',
                                'energy_dependent_gamma_50_eta_50_seed_2_n_500',
                                'energy_dependent_gamma_50_eta_50_seed_3_n_500',
                                'energy_dependent_gamma_50_eta_50_seed_4_n_500',
                                'energy_dependent_gamma_50_eta_50_seed_5_n_500',
                                #'energy_dependent_gamma_50_eta_60_seed_1_n_500',
                                #'energy_dependent_gamma_50_eta_70_seed_1_n_500',
                                             #'energy_dependent_both_seed_3',
                                             #'energy_dependent_both_seed_4',
                                             #'energy_dependent_both_seed_5',
                                             #'energy_dependent_both_seed_6',
                                             #'energy_dependent_both_seed_7',
                                             #'energy_dependent_both_seed_8',
                                             #'energy_dependent_both_seed_9',
                                             #'energy_dependent_both_seed_10',
                                             #'energy_dependent_both_seed_11',
                                             #'energy_dependent_both_seed_12'
                                             ]
else:
    energy_gamma_and_eta_sensitive_config = []

if energy_both_gammaneg:
    energy_gammaneg_and_eta_sensitive_config = [
                                                #'energy_dependent_both_gammaneg_seed_1',
                                                'energy_dependent_neggamma_50_eta_50_k_01_seed_1_n_500'
                                                #'energy_dependent_both_gammaneg_seed_2',
                                                #'energy_dependent_both_gammaneg_seed_3',
                                                #'energy_dependent_both_gammaneg_seed_4',
                                                #'energy_dependent_both_gammaneg_seed_5',
                                                #'energy_dependent_both_gammaneg_seed_6',
                                                #'energy_dependent_both_gammaneg_seed_7',
                                                #'energy_dependent_both_gammaneg_seed_8',
                                                #'energy_dependent_both_gammaneg_seed_9',
                                                #'energy_dependent_both_gammaneg_seed_10'
                                                ]
else:
    energy_gammaneg_and_eta_sensitive_config = []

if energy_both_dis_k:
    energy_gamma_and_eta_sensitive_disrupted_k_config = [
                                    #'energy_dependent_both_dis_k_seed_1',
                                #'energy_dependent_gamma_20_eta_50_k_01_seed_1_n_500',
                                'energy_dependent_gamma_20_eta_50_k_01_seed_2_n_500',
                                'energy_dependent_gamma_20_eta_50_k_01_seed_3_n_500',
                                'energy_dependent_gamma_20_eta_50_k_01_seed_4_n_500',
                                'energy_dependent_gamma_20_eta_50_k_01_seed_5_n_500',
                                #'energy_dependent_gamma_20_eta_50_k_05_seed_1_n_500',
                                'energy_dependent_gamma_20_eta_50_k_05_seed_2_n_500',
                                'energy_dependent_gamma_20_eta_50_k_05_seed_3_n_500',
                                'energy_dependent_gamma_20_eta_50_k_05_seed_4_n_500',
                                'energy_dependent_gamma_20_eta_50_k_05_seed_5_n_500',
                                #'energy_dependent_gamma_20_eta_50_k_07_seed_1_n_500',
                                'energy_dependent_gamma_20_eta_50_k_07_seed_2_n_500',
                                'energy_dependent_gamma_20_eta_50_k_07_seed_3_n_500',
                                'energy_dependent_gamma_20_eta_50_k_07_seed_4_n_500',
                                'energy_dependent_gamma_20_eta_50_k_07_seed_5_n_500',
                                #'energy_dependent_gamma_20_eta_10_k_01_seed_1_n_500',
                                    #'energy_dependent_both_dis_k_seed_2',
                                    #'energy_dependent_both_dis_k_seed_3',
                                                        ]
else:
    energy_gamma_and_eta_sensitive_disrupted_k_config = []

all_config_files = {
    'energy_insensitive_config': energy_insensitive_config,
    'energy_gamma_sensitive_config': energy_gamma_sensitive_config,
    'energy_eta_sensitive_config': energy_eta_sensitive_config,
    'energy_gamma_and_eta_sensitive_config': energy_gamma_and_eta_sensitive_config,
    'energy_gammaneg_and_eta_sensitive_config': energy_gammaneg_and_eta_sensitive_config,
    'energy_gamma_and_eta_sensitive_disrupted_k_config': energy_gamma_and_eta_sensitive_disrupted_k_config
                    }

if __name__ == '__main__':
    start = time.time()
    tot_sims = 0
    for energy_case, energy_case_list in all_config_files.items():
        start_case = time.time()
        print('##############################################################')
        print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
        print('##############################################################')
        print(f'## RUNNING "{energy_case}" CASE ###############')
        print('##############################################################')
        print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
        print('##############################################################')
        for n, file in enumerate(energy_case_list):
            print('######################################################')
            print(f'## RUNNING FILE {n+1}/{len(energy_case_list)} #######')
            print('######################################################')
            print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
            subprocess.run(['python',
                            '-m',
                            'src.experiment',
                            '-f',
                            f'config/{file}.yaml'])
            tot_sims += 1
        end_case = time.time()
        case_time = end_case - start_case
        print('##########################################')
        print('##  CASE COMPUTE TIMES ###################')
        print('##########################################')
        print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
        print(f'case {energy_case} takes: {case_time} sec. -> {case_time/60} min.')
    end = time.time()
    tot_time = end - start
    mean_time = (end - start)/tot_sims
    print('##########################################')
    print('##  ALL CASES COMPUTE TIMES ##############')
    print('##########################################')
    print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
    print(f'all case takes: {tot_time} sec. -> {tot_time/60} min.')
    print(f'In average: {mean_time} sec. -> {mean_time/60} min.')
