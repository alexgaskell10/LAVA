local victim_archive_file = "bin/runs/ruletaker/2021-12-12_17-38-38_roberta-large";
local file_path = ""; 
local pkl_path = "bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05.pkl"; 
local cuda_device = 2;
local attack_method = "textfooler";
local outdir = "bin/runs/baselines";
local outpath = "bin/runs/baselines/textfooler/CustomTextFoolerAttacker_CustomVictim_CustomTokenizer_2021-12-06_22-18-05.txt";

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
