config = {
    "file_path": None, #file_path,
    "dset_config": {
        'add_NAF': False, #True,
        'true_samples_only': False,
        'concat_q_and_c': True,
        'shortest_proof': 0,
        'longest_proof': 100,
        'pretrained_retriever_model': None, #'bin/runs/pretrain_retriever/rb-base/model.tar.gz',
        'retriever_variant': 'roberta-large',
        'sample': -1,
        'use_context_full': False,
        'scramble_context': False,
        'skip_id_regex': '$none',
        'add_prefix': {'c': 'C: ','q': 'Q: '},
        'syntax': 'rulebase',
        'max_pieces': 384,
        'one_proof': False,
        'max_instances': False,
        'pretrained_model': 'roberta-large'
    },
    "archive_config": {
        # "archive_file": archive_file,
        "cuda_device": 3,
        "overrides": ""
    },
    "dataloader_config": {   
        'batches_per_epoch': None,
        'multiprocessing_context': None,
        'worker_init_fn': None,
        'timeout': 0,
        'drop_last': False,
        'pin_memory': False,
        'num_workers': 0,
        'shuffle': False,
        # 'batch_size': batch_size,
    }
}
