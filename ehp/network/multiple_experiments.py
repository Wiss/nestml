"""
multiple_experiments.py
    given a list of config file run multitple experiment
"""
import subprocess
import time

energy_insensitive_config = ['energy_independent_seed_1',
                             'energy_independent_seed_2',
                             'energy_independent_seed_3',
                             ]
energy_gamma_sensitive_config = ['energy_dependent_gamma_seed_1',
                                 'energy_dependent_gamma_seed_2',
                                 'energy_dependent_gamma_seed_3']
energy_eta_sensitive_config = ['energy_dependent_eta_seed_1',
                               'energy_dependent_eta_seed_2',
                               'energy_dependent_eta_seed_3',
                               'energy_dependent_eta_seed_4',
                               'energy_dependent_eta_seed_5',
                               ]
energy_gamma_and_eta_sensitive_config = ['energy_dependent_both_seed_1',
                                         'energy_dependent_both_seed_2',
                                         'energy_dependent_both_seed_3',
                                         'energy_dependent_both_seed_4',
                                         'energy_dependent_both_seed_5',
                                         ]
energy_gamma_and_eta_sensitive_disrupted_k_config = [
                                'energy_dependent_both_dis_k_seed_1',
                                'energy_dependent_both_dis_k_seed_2',
                                'energy_dependent_both_dis_k_seed_3',
                                                     ]

all_config_files = {
    'energy_insensitive_config': energy_insensitive_config,
    'energy_gamma_sensitive_config': energy_gamma_sensitive_config,
    'energy_eta_sensitive_config': energy_eta_sensitive_config,
    'energy_gamma_and_eta_sensitive_config': energy_gamma_and_eta_sensitive_config,
    'energy_gamma_and_eta_sensitive_disrupted_k_config': energy_gamma_and_eta_sensitive_disrupted_k_config
                    }

if __name__ == '__main__':
    start = time.time()
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
        end_case = time.time()
        case_time = end_case - start_case
        print('##########################################')
        print('##  CASE COMPUTE TIMES ###################')
        print('##########################################')
        print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
        print(f'case {energy_case} takes: {case_time} sec. -> {case_time/60} min.')
    end = time.time()
    tot_time = end - start
    print('##########################################')
    print('##  ALL CASES COMPUTE TIMES ##############')
    print('##########################################')
    print('-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o-o')
    print(f'all case takes: {tot_time} sec. -> {tot_time/60} min.')
