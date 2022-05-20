## bash resources/cmds/transferability.sh  
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=8       # Currently only supports single GPU

# Below args for computing attacker transferability
victim1=resources/runs/ruletaker/2021-12-12_17-38-38_roberta-large/
victim2=resources/runs/ruletaker/2021-12-22_09-38-17_distilroberta-base/
adv_dir_1=resources/runs/adversarial/2021-12-12_17-38-38_roberta-large/
adv_dir_2=resources/runs/adversarial/2021-12-12_19-08-47_roberta-base/

dataset_reader_type='records_reader'        # records_reader,baseline_records_reader
suffix='test_results-records.pkl'
ext1=$(echo $victim1 | sed 's+\/$++' | sed -e 's+.*/++')--$(echo $victim2 | sed 's+\/$++' | sed -e 's+.*/++')
# ext2=$(echo $victim2 | sed 's+\/$++' | sed -e 's+.*/++')--$(echo $victim1 | sed 's+\/$++' | sed -e 's+.*/++')

# # Below args for computing baseline transferability (uncomment and adapt as required)
# victim1=resources/runs/ruletaker/2021-12-12_19-08-47_roberta-base/
# victim2=$victim1
# adv_dir_1=resources/runs/baselines/hotflip/
# adv_dir_2=resources/runs/baselines/textfooler/
# dataset_reader_type='baseline_records_reader'        # records_reader,baseline_records_reader
# suffix=2021-12-16_11-00-14_reevaled_bu.pkl
# ext1=rb_lg--rb_base_hotflip
# ext2=rb_lg--rb_base_textfooler

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/transferability_log_'$dt'.out' 2>&1

### 1. Eval victim2 using adversarial samples generated on victim1
raw_config_1=resources/config/transferability/config.jsonnet
proc_config_1=resources/config/transferability/config_$dt.jsonnet
cp $raw_config_1 $proc_config_1

adv_samples_vic1=$adv_dir_1/$suffix
output_file=$victim1'/transferability_results_'$ext1'-2.json'
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1
sed -i 's/local\ dataset_reader_type\ =\ [^;]*;/local\ dataset_reader_type\ =\ "'$dataset_reader_type'";/g' $proc_config_1

echo '\n\nEvaluating the victim 2 ('$victim2') on the adversarial samples from victim 1 ('$victim1').'
echo 'Samples @ '$adv_samples_vic1'. \nOutputs will be saved to '$output_file'\n\n'
cmd='python main.py \ 
        transferability \
        '$victim2'/model.tar.gz \
        '$adv_samples_vic1' \
        --output-file '$output_file' \
        --overrides_file '$proc_config_1' \
        --cuda-device '$cuda_device
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd



### 2. Eval victim1 using adversarial samples generated on victim2
raw_config_2=resources/config/transferability/config.jsonnet
proc_config_2=resources/config/transferability/config_$dt.jsonnet
cp $raw_config_2 $proc_config_2

adv_samples_vic2=$adv_dir_2/$suffix
output_file=$victim2'/transferability_results_'$ext2'-2.json'
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_2
sed -i 's/local\ dataset_reader_type\ =\ [^;]*;/local\ dataset_reader_type\ =\ "'$dataset_reader_type'";/g' $proc_config_1

echo '\n\nEvaluating the victim 1 ('$victim1') on the adversarial samples from victim 2 ('$victim2').'
echo 'Samples @ '$adv_samples_vic2'. \nOutputs will be saved to '$output_file'\n\n'
cmd='python main.py \ 
        transferability \
        '$victim1'/model.tar.gz \
        '$adv_samples_vic2' \
        --output-file '$output_file' \
        --overrides_file '$proc_config_2' \
        --cuda-device '$cuda_device
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

echo 'Done'