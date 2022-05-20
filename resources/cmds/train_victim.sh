## bash resources/cmds/main_flow.sh 
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=3       # Currently only supports single GPU training
roberta_model="roberta-large"        # Victim model
outdir_victim=resources/runs/ruletaker/$dt'_'$roberta_model
outdir_victim_retrain=resources/runs/ruletaker/$dt'_'$roberta_model'_'retrain
outdir_attacker=resources/runs/adversarial/$dt'_'$roberta_model
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/train_victim-'$roberta_model'.out' 2>&1


### STEP 1. Train the victim ###

## Create appropriate config file
raw_config_1=resources/config/ruletaker/rulereasoning_config.jsonnet
proc_config_1=resources/config/ruletaker/rulereasoning_config_$dt.jsonnet
cp $raw_config_1 $proc_config_1

# Set epochs depending on model size
if [ $roberta_model = 'roberta-large' ]
then
  num_epochs=8 #4
  batch_size=2
  num_gradient_accumulation_steps=8
  learning_rate=5e-6 #1e-5
else
  num_epochs=16 #8
  batch_size=8
  num_gradient_accumulation_steps=2
  learning_rate=1e-5
fi
dropout=-1

sed -i 's/local\ cuda_device\ =\ [[:digit:]]\+/local\ cuda_device\ =\ '$cuda_device'/g' $proc_config_1
sed -i 's/local\ num_epochs\ =\ [[:digit:]]\+/local\ num_epochs\ =\ '$num_epochs'/g' $proc_config_1
sed -i 's/local\ batch_size\ =\ [[:digit:]]\+/local\ batch_size\ =\ '$batch_size'/g' $proc_config_1
sed -i 's/local\ num_gradient_accumulation_steps\ =\ [[:digit:]]\+/local\ num_gradient_accumulation_steps\ =\ '$num_gradient_accumulation_steps'/g' $proc_config_1
sed -i 's/local\ transformer_model\ \=\ [^,]*;/local\ transformer_model\ =\ "'$roberta_model'";/g' $proc_config_1
sed -i 's+local\ dataset_dir\ \=\ [^,]*;+local\ dataset_dir\ =\ "'$data_dir'";+g' $proc_config_1
sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1
sed -i 's+local\ learning_rate\ \=\ [^;]*;+local\ learning_rate\ =\ '$learning_rate';+g' $proc_config_1
sed -i 's+local\ dropout\ \=\ [^;]*;+local\ dropout\ =\ '$dropout';+g' $proc_config_1

# Train the model
echo '\n\nTraining the victim model using config '$proc_config_1'. \nOutputs will be saved to '$outdir_victim'\n\n'
cmd='python main.py \
        ruletaker_train_original \
        '$proc_config_1' \
        -s '$outdir_victim
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd

# Evaluate trained model on the test set
echo '\n\nEvaluating the victim model on the test set (no config file). \nOutputs will be saved to '$outdir_victim/test_results.json'\n\n'
overrides='{"trainer":{"cuda_device":"'$cuda_device'"},"validation_data_loader":{"batch_sampler":{"batch_size":64,"type":"bucket"}}}'
cmd='python main.py \
        ruletaker_test_original \
        '$outdir_victim'/model.tar.gz \
        '$data_dir'/test.jsonl \
        --output-file '$outdir_victim'/test_results.json \
        --cuda-device '$cuda_device' \
        -o "'$overrides'"'
cmd=$(sed 's/\\//g' <<< $cmd)
echo $cmd
$cmd
