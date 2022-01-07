local victim_archive_file = "bin/runs/ruletaker/2021-12-20_10-48-09_roberta-large";
local file_path = ""; 
local pkl_path = "bin/runs/baselines/textfooler/2021-12-21_18-12-03.pkl"; 
local cuda_device = 5;
local attack_method = "textfooler";
local outdir = "bin/runs/baselines";
local outpath = "bin/runs/baselines/textfooler/2021-12-21_18-12-03.pkl";

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
