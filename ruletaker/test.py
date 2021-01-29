import os

a = 0
if a:
    import wandb
    wandb.init(project='test')
    os.environ['WANDB_LOG'] = 'true'
else:
    os.environ['WANDB_LOG'] = 'false'

if os.environ['WANDB_LOG'] == 'true':
    wandb.log({'test': 1})

