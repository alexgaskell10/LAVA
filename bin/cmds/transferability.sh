## bash bin/cmds/transferability.sh  
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=7       # Currently only supports single GPU
victim1=bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/
victim2=bin/runs/ruletaker/2021-12-12_19-08-47_roberta-base/
adv_dir_1=bin/runs/adversarial/2021-12-12_17-38-38_roberta-large/
adv_dir_2=bin/runs/adversarial/2021-12-12_19-08-47_roberta-base/


exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/transferability_log.out' 2>&1


### 1. Eval victim2 using adversarial samples generated on victim1
raw_config_1=bin/config/transferability/config.jsonnet
proc_config_1=bin/config/transferability/config_$dt.jsonnet
cp $raw_config_1 $proc_config_1

adv_samples_vic1=$adv_dir_1/test_results-records.pkl
ext=$(echo $victim1 | sed 's+\/$++' | sed -e 's+.*/++')--$(echo $victim2 | sed 's+\/$++' | sed -e 's+.*/++')
output_file=$victim1'/transferability_results_'$ext'.json'
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1

echo '\n\nEvaluating the victim 2 ('$victim2') on the adversarial samples from victim 1 ('$victim1').'
echo 'Samples @ '$adv_samples_vic1'. \nOutputs will be saved to '$output_file'\n\n'
cmd='python main.py \ 
        transferability \
        '$victim2'/model.tar.gz \
        '$adv_samples_vic1' \
        --output-file '$output_file' \
        --overrides_file '$proc_config_1' \
        --cuda-device '$cuda_device' \
        --include-package ruletaker.allennlp_models'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd



### 2. Eval victim1 using adversarial samples generated on victim2
raw_config_2=bin/config/transferability/config.jsonnet
proc_config_2=bin/config/transferability/config_$dt.jsonnet
cp $raw_config_2 $proc_config_2

adv_samples_vic2=$adv_dir_2/test_results-records.pkl
ext=$(echo $victim2 | sed 's+\/$++' | sed -e 's+.*/++')--$(echo $victim1 | sed 's+\/$++' | sed -e 's+.*/++')
output_file=$victim2'/transferability_results_'$ext'.json'
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2

echo '\n\nEvaluating the victim 1 ('$victim1') on the adversarial samples from victim 2 ('$victim2').'
echo 'Samples @ '$adv_samples_vic2'. \nOutputs will be saved to '$output_file'\n\n'
cmd='python main.py \ 
        transferability \
        '$victim1'/model.tar.gz \
        '$adv_samples_vic2' \
        --output-file '$output_file' \
        --overrides_file '$proc_config_2' \
        --cuda-device '$cuda_device' \
        --include-package ruletaker.allennlp_models'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

echo 'Done'