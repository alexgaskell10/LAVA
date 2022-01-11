# dt=$(date +%Y-%m-%d_%H-%M-%S)
cuda_device=9
max_instances=-1
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/

# dt='2021-12-12_17-38-38'
# ext=$dt'_roberta-large'

dt='2021-12-12_19-08-47'
ext=$dt'_roberta-base'

outdir_victim=resources/runs/ruletaker/$ext
outdir_attacker=resources/runs/adversarial/$ext

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