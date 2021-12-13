## bash bin/cmds/transferability.sh  
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=2       # Currently only supports single GPU
victim='bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large'
data='data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl'
outdir='bin/runs/baselines'
outpath=$outdir/

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/hf_tf_baselines_log.out' 2>&1


for attack_method in 'hotflip' 'textfooler'
do
    ### 1. Run the attack method on the victim over the test set and generate predictions
    ext=$attack_method'_'$dt
    raw_config_1=bin/config/baselines/config.jsonnet
    proc_config_1=bin/config/baselines/config_$ext.jsonnet
    cp $raw_config_1 $proc_config_1
    outpath=$outdir/$ext'.pkl'

    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1
    sed -i 's/local\ cuda_device\ =\ [^;]*;/local\ cuda_device\ =\ '$cuda_device';/g' $proc_config_1
    sed -i 's/local\ attack_method\ =\ [^;]*;/local\ attack_method\ =\ "'$attack_method'";/g' $proc_config_1
    sed -i 's+local\ victim_archive_file\ =\ [^;]*;+local\ victim_archive_file\ =\ "'$victim'";+g' $proc_config_1
    sed -i 's+local\ file_path\ =\ [^;]*;+local\ file_path\ =\ "'$data'";+g' $proc_config_1
    sed -i 's+local\ outdir\ =\ [^;]*;+local\ outdir\ =\ "'$outdir'";+g' $proc_config_1
    sed -i 's+local\ outpath\ =\ [^;]*;+local\ outpath\ =\ "'$outpath'";+g' $proc_config_1

    echo '\n\nAttacking the victim ('$victim') using '$attack_method'. \nOutputs will be saved to '$outpath'\n\n'
    cmd='python adversarial/openattack/openattack.py '$proc_config_1
    echo $cmd
    $cmd




    ### 2. Re-evaluate the modified logic program using the problog solver to compute the revised labels and results
    proc_config_2=bin/config/baselines/config_$ext'_reeval.jsonnet'
    ext=$attack_method'_'$dt'_reeval'
    cp $proc_config_1 $proc_config_2
    attack_records=$outpath
    outpath=$(sed 's+\.pkl+\_modresults\.txt+g' <<< $attack_records)

    # Use the outputs from the above job as the input for this job
    sed -i 's+local\ pkl_path\ =\ [^;]*;+local\ pkl_path\ =\ "'$attack_records'";+g' $proc_config_2
    sed -i 's+local\ file_path\ =\ [^;]*;+local\ file_path\ =\ "";+g' $proc_config_2

    echo '\n\nComputing the revised entailment relationships and attack flip rates. \nOutputs will be saved to '$outpath'\n\n'
    cmd='python adversarial/openattack/reeval.py '$proc_config_2
    echo $cmd
    $cmd

    echo 'Done'
done