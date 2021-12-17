max_instances=-1        # 10 -1
cuda_device=9       # Currently only supports single GPU training
roberta_model="roberta-large"        # Victim model
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/

dt='2021-12-12_17-38-38'
outdir_victim=bin/runs/ruletaker/$dt'_'$roberta_model
outdir_victim_retrain=bin/runs/ruletaker/$dt'_'$roberta_model'_retrain'
outdir_attacker=bin/runs/adversarial/$dt'_'$roberta_model


exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/retrain-'$roberta_model'.out' 2>&1



### STEP 3. Retrain the victim on the combined dataset ###

raw_config_4=bin/config/ruletaker/ruletaker_adv_retraining.jsonnet
proc_config_4=bin/config/ruletaker/ruletaker_adv_retraining_$dt.jsonnet
cp $raw_config_4 $proc_config_4

# Set epochs depending on model size
if [ $roberta_model = 'roberta-large' ]
then
  num_epochs=4
  batch_size=2
  num_gradient_accumulation_steps=8
else
  num_epochs=8
  batch_size=4
  num_gradient_accumulation_steps=4
fi

sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_4
sed -i 's/local\ num_epochs\ =\ [[:digit:]]\+/local\ num_epochs\ =\ '$num_epochs'/g' $proc_config_4
sed -i 's/local\ batch_size\ =\ [[:digit:]]\+/local\ batch_size\ =\ '$batch_size'/g' $proc_config_4
sed -i 's/local\ num_gradient_accumulation_steps\ =\ [[:digit:]]\+/local\ num_gradient_accumulation_steps\ =\ '$num_gradient_accumulation_steps'/g' $proc_config_4
sed -i 's/local\ transformer\_model\ \=\ [^,]*;/local\ transformer\_model\ =\ "'$roberta_model'";/g' $proc_config_4
sed -i 's+local\ dataset_dir\ \=\ [^,]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_4
sed -i 's+local\ transformer_weights_model\ \=\ [^,]*;+local\ transformer_weights_model\ =\ "'$outdir_victim'";+g' $proc_config_4

# Get path for adversarial examples
path=$(ls $outdir_attacker | grep val-records_epoch-1)
if [ -z "$path" ]
then
      adv_train_path=$outdir_attacker/$(ls $outdir_attacker | grep train-records_epoch | tail -1)
else
      adv_train_path=$outdir_attacker/$path
fi

adv_val_path=$outdir_attacker/$(ls $outdir_attacker | grep val-records_epoch | tail -1)
adv_test_path=$outdir_attacker/$(ls $outdir_attacker | grep test-records_epoch | tail -1)

sed -i 's+local\ adversarial_examples_path_train\ \=\ [^,]*;+local\ adversarial_examples_path_train\ =\ "'$adv_train_path'";+g' $proc_config_4
sed -i 's+local\ adversarial_examples_path_val\ \=\ [^,]*;+local\ adversarial_examples_path_val\ =\ "'$adv_val_path'";+g' $proc_config_4
sed -i 's+local\ adversarial_examples_path_test\ \=\ [^,]*;+local\ adversarial_examples_path_test\ =\ "'none'";+g' $proc_config_4
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_4

# Continue training model on augmented dataset
echo '\n\nRetraining the victim model on the augmented dataset using config '$proc_config_4'. \nOutputs will be saved to '$outdir_victim_retrain'\n\n'
cmd='python main.py \
        ruletaker_adv_training \
        '$proc_config_4' \
        -s '$outdir_victim_retrain' \
        --include-package ruletaker.allennlp_models'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

# Evaluate trained model on the test set (original, adv, combined)
raw_config_5=bin/config/ruletaker/ruletaker_adv_retraining_test.jsonnet
proc_config_5=bin/config/ruletaker/ruletaker_adv_retraining_test_$dt.jsonnet
cp $raw_config_5 $proc_config_5

sed -i 's+local\ dataset_dir\ \=\ [^,]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_5
sed -i 's+local\ adversarial\_examples_path_test\ \=\ [^,]*;+local\ adversarial_examples_path_test\ =\ "'$adv_test_path'";+g' $proc_config_5
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_5

echo '\n\nEvaluating the victim on the augmented test set using config '$proc_config_5'. \nOutputs will be saved to '$outdir_victim_retrain'\n\n'
cmd='python main.py \
        ruletaker_adv_training_test \
        '$outdir_victim_retrain'/model.tar.gz \
        test \
        --output-file '$outdir_victim_retrain'/aug_test_results.json \
        --overrides_file '$proc_config_5' \
        --cuda-device '$cuda_device' \
        --include-package ruletaker.allennlp_models'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

echo "Done"
