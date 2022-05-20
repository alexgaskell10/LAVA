## bash resources/cmds/main_flow.sh 
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=9
outdir_victim=resources/runs/ruletaker/2021-12-12_17-38-38_roberta-large
outdir_attacker=resources/runs/adversarial/2021-12-12_17-38-38_roberta-large
data_dir=data/rule-reasoning-dataset-V2020.2.4/depth-5/

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/val_mc.out' 2>&1


for val_num_monte_carlo in 1 2 4 8 16
do
    # Eval trained attacker on the test set
    raw_config=resources/config/attacker/test_config.jsonnet
    proc_config=resources/config/attacker/test_config_$dt.jsonnet
    cp $raw_config $proc_config

    sed -i 's+local\ ruletaker_archive\ =\ [^,]*;+local\ ruletaker_archive\ =\ '"'"$outdir_victim/model.tar.gz"'"';+g' $proc_config
    sed -i 's+local\ val_num_monte_carlo\ =\ [^,]*;+local\ val_num_monte_carlo\ =\ '$val_num_monte_carlo';+g' $proc_config
    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config

    echo '\n\nEvaluating the attacker on the test set using config '$proc_config'. \nOutputs will be saved to '$outdir_attacker'\n\n'
    cmd='python main.py \
            adversarial_dataset_generation_test \
            '$outdir_attacker'/model.tar.gz \
            '$data_dir'/test.jsonl \
            --output-file '$outdir_attacker'/val_mc-'$val_num_monte_carlo'_test_results.json \
            --overrides_file '$proc_config' \
            --cuda-device '$cuda_device
    cmd=$(sed 's/\\//g' <<< $cmd)
    echo $cmd
    $cmd
done
