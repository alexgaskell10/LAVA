local victim_archive_file = "bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large";
local file_path = "data/rule-reasoning-dataset-V2020.2.4/depth-5/train.jsonl"; 
local pkl_path = ""; 
local cuda_device = 8;
local attack_method = "hotflip";
local outdir = "bin/runs/baselines";
local outpath = "bin/runs/baselines/hotflip/2021-12-20_19-07-51_reeval.txt";

{
    "outpath": outpath,
    "file_path": file_path,
    "pkl_path": pkl_path,
    "max_instances": -1,
    "victim_archive_file": victim_archive_file,
    "attacker": attack_method,
    "cuda_device": cuda_device,
    "outdir": outdir,
}
