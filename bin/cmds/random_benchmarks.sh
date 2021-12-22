## bash bin/cmds/transferability.sh  
dt=$(date +%Y-%m-%d_%H-%M-%S)

max_instances=-1        # 10 -1
cuda_device=8       # Currently only supports single GPU
victim='bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large/'
data='data/rule-reasoning-dataset-V2020.2.4/depth-5/test.jsonl'
outdir='bin/runs/baselines/random_adversarial/'
# outpath=$outdir/

exec 3>&1 4>&2
trap 'exec 2>&4 1>&3' 0 1 2 3
exec 1>'logs/random_benchmarks_log.out' 2>&1




for benchmark_variation in 'random' 'word_score'
do
    ext=$benchmark_variation'_'$dt
    raw_config_1=bin/config/baselines/adversarial_benchmark/config.jsonnet
    proc_config_1=bin/config/baselines/adversarial_benchmark/config_$ext.jsonnet
    cp $raw_config_1 $proc_config_1

    sed -i 's/"max_instances":\ [^,]*,/"max_instances":\ '$max_instances',/g' $proc_config_1
    sed -i 's+local\ victim_archive_file\ =\ [^;]*;+local\ victim_archive_file\ =\ "'$victim'";+g' $proc_config_1
    sed -i 's+local\ benchmark\_variation\ =\ [^;]*;+local\ benchmark\_variation\ =\ "'$benchmark_variation'";+g' $proc_config_1

    echo '\n\nAttacking the victim ('$victim') using the '$benchmark_variation' benchmark. \nOutputs will be saved to '$outdir/$ext'\n\n'
    cmd='python main.py \
            adversarial_random_benchmark \
            ""
            '$data' \
            --output-file '$outdir/$ext'_results.json \
            --overrides_file '$proc_config_1' \
            --cuda-device '$cuda_device' \
            --fresh-init \
            --include-package ruletaker.allennlp_models'
    cmd=$(sed 's/\\//g' <<< $cmd)
    echo $cmd
    $cmd
done
echo 'Done'
