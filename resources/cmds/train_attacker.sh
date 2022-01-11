## bash resources/cmds/main_flow.sh 
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=5       # Currently only supports single GPU training

roberta_model="roberta-large"        # Victim model
outdir_attacker=resources/runs/adversarial/$dt'_'$roberta_model'_multinom'
outdir_victim=resources/runs/ruletaker/2022-01-10_08-25-19_roberta-large
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/train_attacker-'$roberta_model'.out' 2>&1



## Create appropriate config file
raw_config_2=resources/config/attacker/config.jsonnet
proc_config_2=resources/config/attacker/config_$dt.jsonnet
cp $raw_config_2 $proc_config_2

sed -i 's+local\ ruletaker_archive\ =\ [^;]*;+local\ ruletaker_archive\ =\ '"'"$outdir_victim/model.tar.gz"'"';+g' $proc_config_2
sed -i 's+local\ dataset_dir\ =\ [^;]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_2
sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_2
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

# Train model
echo '\n\nTraining the attacker model using config '$proc_config_2'. \nOutputs will be saved to '$outdir_attacker'\n\n'
cmd='python main.py \
        adversarial_dataset_generation \
        '$proc_config_2' \
        -s '$outdir_attacker' \
        --include-package lava'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

# Eval trained attacker on the test set
raw_config_3=resources/config/attacker/test_config.jsonnet
proc_config_3=resources/config/attacker/test_config_$dt.jsonnet
cp $raw_config_3 $proc_config_3

sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$outdir_victim/model.tar.gz"'"';+g' $proc_config_3
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_3

echo '\n\nEvaluating the attacker on the test set using config '$proc_config_3'. \nOutputs will be saved to '$outdir_attacker'\n\n'
cmd='python main.py \
        adversarial_dataset_generation_test \
        '$outdir_attacker'/model.tar.gz \
        '$data_dir'/test.jsonl \
        --output-file '$outdir_attacker'/test_results.json \
        --overrides_file '$proc_config_3' \
        --cuda-device '$cuda_device' \
        --include-package lava'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

