## bash resources/cmds/hf_tf_baselines.sh
dt=$(date +%Y-%m-%d_%H-%M-%S)

### USER INPUT (BEGIN) ###
max_instances=-1        # 10 -1
cuda_device=9       # Currently only supports single GPU
dset=test
victim='resources/runs/ruletaker/2022-01-10_08-25-19_roberta-large'
data='data/rule-reasoning-dataset-V2020.2.4/depth-5/'$dset'.jsonl'
outdir='resources/runs/baselines'
outpath=$outdir/
### USER INPUT (END) ###

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/hf_tf_baselines_log.out' 2>&1


for attack_method in 'hotflip' 'textfooler'
do
    ### 1. Run the attack method on the victim over the data and generate predictions
    ext=$attack_method/$dt'_'$dset
    raw_config_1=resources/config/baselines/config.jsonnet
    proc_config_1=resources/config/baselines/$ext'_config.jsonnet'
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
    cmd='python lava/openattack/openattack.py '$proc_config_1
    echo $cmd
    $cmd




    ### 2. Re-evaluate the modified logic program using the problog solver to compute the revised labels and results
    proc_config_2=resources/config/baselines/$ext'_reeval.jsonnet'
    cp $proc_config_1 $proc_config_2
    attack_records=$outdir/$ext'.pkl'
    ext_=$attack_method/$dt'_'$dset'_reeval'
    outpath2=$outdir/$ext_'.txt'

    # Use the outputs from the above job as the input for this job
    sed -i 's+local\ pkl_path\ =\ [^;]*;+local\ pkl_path\ =\ "'$attack_records'";+g' $proc_config_2
    sed -i 's+local\ file_path\ =\ [^;]*;+local\ file_path\ =\ "";+g' $proc_config_2
    sed -i 's+local\ outpath\ =\ [^;]*;+local\ outpath\ =\ "'$outpath2'";+g' $proc_config_2

    echo '\n\nComputing the revised entailment relationships and attack flip rates. \nOutputs will be saved to '$outpath'\n\n'
    cmd='python lava/openattack/reeval.py '$proc_config_2
    echo $cmd
    $cmd

    echo 'Done'
done