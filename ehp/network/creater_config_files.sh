#!/bin/bash

GENERATE_CONFIG_FILES=1
MODIFY_CONFIG_FILE=1

# generate copies from file <SOURCE_FILE> to <TARGET_FILE>
if [ "$GENERATE_CONFIG_FILES" == 1 ]
then
    SOURCE_FILE="energy_dependent_gamma_20_eta_50_k_07_seed_1_n_500"
    TARGET_FILE="energy_dependent_gamma_20_eta_50_k_07_seed_" 
    for j in {2..5}
        # copy from one source file 
        do cp config/$SOURCE_FILE.yaml config/$TARGET_FILE"$j""_n_500".yaml
        # copy from several source files
        #do cp config/$SOURCE_FILE"$j".yaml config/$TARGET_FILE"$j".yaml
    done
fi

# modify parameters from config files
if [ "$MODIFY_CONFIG_FILE" == 1 ]
then
    # seed -> 15 (2spaces seed: value)
    # gamma_ex -> 68 (10spaces mean: value)
    # gamma_in -> 141 (10spaces mean: value)
    # I_e ex -> 92 (10spaces mean: value)
    # I_e in -> 165 (10spaces mean: value)
    # eta_ex_ex -> 185 (8spaces eta: value)
    # new value for the paameters
    NEW_VALUE="250"

    declare -a PARAM_TO_MODIFY=(14)  # for seed

    #declare -a PARAM_TO_MODIFY=(92 165)  # for I_e
    NEW_TEXT="          mean: $NEW_VALUE"  # for I_e 

    #declare -a PARAM_TO_MODIFY=(185)  # for eta
    #NEW_TEXT="        eta: $NEW_VALUE"  # for eta 

    #declare -a PARAM_TO_MODIFY=(68 141)  # for gamma
    #NEW_TEXT="          mean: $NEW_VALUE"  # for gamma

    FILE="energy_dependent_gamma_20_eta_50_k_07_seed_"
    for n in {2..5}
        do for m in ${PARAM_TO_MODIFY[@]}
            do 
            if [[ "$m" -eq "14" ]]
            then
                NEW_TEXT="  seed: $n"  # for seed
            fi
            echo $m
            echo $NEW_TEXT
            sed -i "$m"'s/.*/'"$NEW_TEXT"'/g' config/$FILE"$n""_n_500".yaml
        done
    done
fi
