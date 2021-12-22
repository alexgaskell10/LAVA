## bash bin/cmds/main_flow.sh 
dt=$(date +%Y-%m-%d_%H-%M-%S)

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/ablate_'$dt'.out' 2>&1


max_instances=-1        # 10 -1
cuda_device=7       # Currently only supports single GPU training
victim=bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/
outdir_attacker_base=bin/runs/ablate/$dt
mkdir -p $outdir_attacker_base
config_dir=bin/config/ablate/$dt
mkdir -p $config_dir

for perturbs in 'equivalence_substitution,question_flip' 'sentence_elimination,question_flip' 'sentence_elimination,equivalence_substitution'
# for perturbs in 'equivalence_substitution' 'question_flip' 'sentence_elimination' 'equivalence_substitution,question_flip,sentence_elimination'

do
    ## Create appropriate config file
    raw_config_1=bin/config/ablate/config.jsonnet
    proc_config_1=$config_dir/$perturbs'_config.jsonnet'
    cp $raw_config_1 $proc_config_1

    outdir_attacker=$outdir_attacker_base/$perturbs/

    sed -i 's+local\ ruletaker_archive\ =\ [^;]*;+local\ ruletaker_archive\ =\ '"'"$victim/model.tar.gz"'"';+g' $proc_config_1
    sed -i 's+local\ dataset_dir\ =\ [^;]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_1
    sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_1
    sed -i 's+local\ adversarial_perturbations\ =\ [^;]*;+local\ adversarial_perturbations\ =\ "'$perturbs'";+g' $proc_config_1
    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1

    # Train model
    echo '\n\nTraining the attacker model using config '$proc_config_1'. \nOutputs will be saved to '$outdir_attacker'\n\n'
    cmd='python main.py \
            adversarial_dataset_generation \
            '$proc_config_1' \
            -s '$outdir_attacker' \
            --include-package ruletaker.allennlp_models'
    cmd=$(sed 's/\\//g' <<< $cmd)
    echo $cmd
    $cmd


    # Eval trained attacker on the test set
    raw_config_2=bin/config/ablate/test_config.jsonnet
    proc_config_2=$config_dir/$perturbs'_test_config_'$dt'.jsonnet'
    cp $raw_config_2 $proc_config_2

    sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$victim/model.tar.gz"'"';+g' $proc_config_2
    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

    echo '\n\nEvaluating the attacker on the test set using config '$proc_config_2'. \nOutputs will be saved to '$outdir_attacker'\n\n'
    cmd='python main.py \
            adversarial_dataset_generation_test \
            '$outdir_attacker'/model.tar.gz \
            '$data_dir'/test.jsonl \
            --output-file '$outdir_attacker'/test_results.json \
            --overrides_file '$proc_config_2' \
            --cuda-device '$cuda_device' \
            --include-package ruletaker.allennlp_models'
    cmd=$(sed 's/\\//g' <<< $cmd)
    echo $cmd
    $cmd
done
echo "Done"











# max_instances=-1        # 10 -1
# cuda_device=5       # Currently only supports single GPU training
# victim=bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large
# data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/
# outdir_attacker_base=bin/runs/ablate/$dt
# mkdir -p $outdir_attacker_base
# config_dir=bin/config/ablate/$dt
# mkdir -p $config_dir

# # for perturbs in 'equivalence_substitution,question_flip' 'sentence_elimination,question_flip' 'sentence_elimination,equivalence_substitution'
# for perturbs in 'equivalence_substitution' 'question_flip' 'equivalence_substitution,question_flip'

# do
#     ## Create appropriate config file
#     raw_config_1=bin/config/ablate/config.jsonnet
#     proc_config_1=$config_dir/$perturbs'_config.jsonnet'
#     cp $raw_config_1 $proc_config_1

#     outdir_attacker=$outdir_attacker_base/$perturbs/

#     sed -i 's+local\ ruletaker_archive\ =\ [^;]*;+local\ ruletaker_archive\ =\ '"'"$victim/model.tar.gz"'"';+g' $proc_config_1
#     sed -i 's+local\ dataset_dir\ =\ [^;]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_1
#     sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_1
#     sed -i 's+local\ adversarial_perturbations\ =\ [^;]*;+local\ adversarial_perturbations\ =\ "'$perturbs'";+g' $proc_config_1
#     sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1

#     # Train model
#     echo '\n\nTraining the attacker model using config '$proc_config_1'. \nOutputs will be saved to '$outdir_attacker'\n\n'
#     cmd='python main.py \
#             adversarial_dataset_generation \
#             '$proc_config_1' \
#             -s '$outdir_attacker' \
#             --include-package ruletaker.allennlp_models'
#     cmd=$(sed 's/\\//g' <<< $cmd)
#     echo $cmd
#     $cmd


#     # Eval trained attacker on the test set
#     raw_config_2=bin/config/ablate/test_config.jsonnet
#     proc_config_2=$config_dir/$perturbs'_test_config_'$dt'.jsonnet'
#     cp $raw_config_2 $proc_config_2

#     sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$victim/model.tar.gz"'"';+g' $proc_config_2
#     sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

#     echo '\n\nEvaluating the attacker on the test set using config '$proc_config_2'. \nOutputs will be saved to '$outdir_attacker'\n\n'
#     cmd='python main.py \
#             adversarial_dataset_generation_test \
#             '$outdir_attacker'/model.tar.gz \
#             '$data_dir'/test.jsonl \
#             --output-file '$outdir_attacker'/test_results.json \
#             --overrides_file '$proc_config_2' \
#             --cuda-device '$cuda_device' \
#             --include-package ruletaker.allennlp_models'
#     cmd=$(sed 's/\\//g' <<< $cmd)
#     echo $cmd
#     $cmd
# done
# echo "Done"