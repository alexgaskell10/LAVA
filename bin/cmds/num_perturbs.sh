## bash bin/cmds/num_perturbs.sh 
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=6       # Currently only supports single GPU training
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/
victim_dir=bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/
outdir_attacker_base=bin/runs/num_perturbs/
mkdir -p $outdir_attacker_base

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/num_perturbs-log-'$dt'.out' 2>&1

base_dir=bin/runs/num_perturbs
for dir in $(ls $outdir_attacker_base)
do
    for run in $(ls $outdir_attacker_base/$dir)
    do
        ext=$run
        outdir_attacker=$outdir_attacker_base/$dir/$ext
        outdir_config=bin/config/num_perturbs/$dt
        mkdir -p $outdir_attacker
        mkdir -p $outdir_config

        # Eval trained attacker on the test set
        raw_config_2=bin/config/num_perturbs/test_config.jsonnet
        proc_config_2=bin/config/num_perturbs/$dt/$ext.jsonnet
        cp $raw_config_2 $proc_config_2

        sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$victim_dir/model.tar.gz"'"';+g' $proc_config_2
        sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

        echo '\n\nEvaluating the attacker on the test set using config '$proc_config_2' with '$ext'. \nOutputs will be saved to '$outdir_attacker'\n\n'
        cmd='python main.py \
                adversarial_dataset_generation_test \
                '$outdir_attacker'/model.tar.gz \
                '$data_dir'/test.jsonl \
                --output-file '$outdir_attacker'/test_results_2.json \
                --overrides_file '$proc_config_2' \
                --cuda-device '$cuda_device' \
                --include-package ruletaker.allennlp_models'
        cmd=$(sed 's/\\//g' <<< $cmd)
        echo $cmd
        $cmd
        
    done
done


# max_instances=-1        # 10 -1
# cuda_device=6       # Currently only supports single GPU training
# data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/
# victim_dir=bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/
# outdir_attacker_base=bin/runs/num_perturbs/$dt/
# mkdir -p $outdir_attacker_base

# exec 3>&1 4>&2
# trap 'exec 2>&4 1>&3' 0 1 2 3
# exec 1>'logs/num_perturbs-log-'$dt'.out' 2>&1

# for ES in {1..5}
# do
#     for SE in {1..5}
#     do
#         ext=SE-$SE'_'ES-$ES
#         outdir_attacker=$outdir_attacker_base/$ext
#         outdir_config=bin/config/num_perturbs/$dt
#         mkdir -p $outdir_attacker
#         mkdir -p $outdir_config

#         ## Create appropriate config file
#         raw_config_1=bin/config/num_perturbs/config.jsonnet
#         proc_config_1=$outdir_config/$ext.jsonnet
#         cp $raw_config_1 $proc_config_1

#         sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$victim_dir/model.tar.gz"'"';+g' $proc_config_1
#         sed -i 's+local\ dataset_dir\ =\ [^,]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_1
#         sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_1
#         sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1
#         sed -i 's+local\ max_flips\ =\ [^,]*;+local\ max_flips\ =\ "'$ES'";+g' $proc_config_1
#         sed -i 's+local\ max_elims\ =\ [^,]*;+local\ max_elims\ =\ "'$SE'";+g' $proc_config_1

#         # Train model
#         echo '\n\nTraining the attacker model using config '$proc_config_1' with '$ext'. \nOutputs will be saved to '$outdir_attacker'\n\n'
#         cmd='python main.py \
#                 adversarial_dataset_generation \
#                 '$proc_config_1' \
#                 -s '$outdir_attacker' \
#                 --include-package ruletaker.allennlp_models'
#         cmd=$(sed 's/\\//g' <<< $cmd)
#         echo $cmd
#         $cmd


#         # Eval trained attacker on the test set
#         raw_config_2=bin/config/num_perturbs/test_config.jsonnet
#         proc_config_2=bin/config/num_perturbs/$dt/$ext.jsonnet
#         cp $raw_config_2 $proc_config_2

#         sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$victim_dir/model.tar.gz"'"';+g' $proc_config_2
#         sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

#         echo '\n\nEvaluating the attacker on the test set using config '$proc_config_2' with '$ext'. \nOutputs will be saved to '$outdir_attacker'\n\n'
#         cmd='python main.py \
#                 adversarial_dataset_generation_test \
#                 '$outdir_attacker'/model.tar.gz \
#                 '$data_dir'/test.jsonl \
#                 --output-file '$outdir_attacker'/test_results.json \
#                 --overrides_file '$proc_config_2' \
#                 --cuda-device '$cuda_device' \
#                 --include-package ruletaker.allennlp_models'
#         cmd=$(sed 's/\\//g' <<< $cmd)
#         echo $cmd
#         $cmd
#     done
# done
