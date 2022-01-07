## bash bin/cmds/hf_tf_baselines.sh
# dt=$(date +%Y-%m-%d_%H-%M-%S)
dt=2021-12-21_18-12-03

max_instances=-1        # 10 -1
cuda_device=8       # Currently only supports single GPU
dset=train
victim='bin/runs/ruletaker/2021-12-21_08-55-58_roberta-base'
data='data/rule-reasoning-dataset-V2020.2.4/depth-5/'$dset'.jsonl'
outdir='bin/runs/baselines'

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/hf_tf_baselines_transferability_'$dt'.out' 2>&1


for attack_method in hotflip textfooler

do
    adv_samples_vic1=$outdir/$attack_method/$dt'_reevaled.pkl'
    
    ### Evalute the new victim on the existing adversarial examples
    raw_config_1=bin/config/baselines/transferability/config.jsonnet
    proc_config_1=bin/config/baselines/transferability/config_$dt.jsonnet
    cp $raw_config_1 $proc_config_1

    output_file=$outdir/$attack_method'/transferability_results_'$dt'.json'
    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1

    echo '\n\nEvaluating the victim ('$victim') on the '$attack_method' adversarial samples.'
    echo 'Samples @ '$adv_samples_vic1'. \nOutputs will be saved to '$output_file'\n\n'
    cmd='python main.py \ 
            transferability \
            '$victim'/model.tar.gz \
            '$adv_samples_vic1' \
            --output-file '$output_file' \
            --overrides_file '$proc_config_1' \
            --cuda-device '$cuda_device' \
            --include-package ruletaker.allennlp_models'
    cmd=$(sed 's/\\//g' <<< $cmd)
    echo $cmd
    $cmd
done
echo "Done"