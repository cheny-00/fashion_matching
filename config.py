import copy
params = {
    "train_batch_size": 64,
    "lr": 3e-5,
    "test_batch_size": 128,
    "device": 'cuda',
    "epochs" : 120,
    "use_tensorboard": True,
    "log_interval": 100,
    "d_model": 2048,
    "ckpt_prefix": "checkpoints/df_",
    "ckpt_save_path": "checkpoints/df.pt",
    "n_instances" : 4,
    "margin": 0.5,
    "dist_fn": "euclidean",
    "query_contrastive_weight": 1.0,
    "query_xent_weight": 1.0,
    "query_center_weight": 0.0005,
    "center_lr": 0.5,
    "do_warmup": True,
    "n_warmup_step": 10, # epoch
    "base_lr": 1e-4,
    "num_classes": 10393,
    "do_resample": False,
    "model_name": 'embed_model',
    "reuse_checkpoint": True,
    "load_ckpt_path": "past_checkpoints/df_50.pt",
    
    
}

# fine_tune_params = copy.deepcopy(params)
# fine_tune_params['train_batch_size'] = 64
# fine_tune_params['epochs'] = 30
# fine_tune_params['ckpt_prefix'] = "checkpoints/ft_df_"
# fine_tune_params['ckpt_save_path'] = "checkpoints/ft_df.pt"
# fine_tune_params['do_warmup'] = True
# fine_tune_params["reuse_checkpoint"] = True



dataset_path = {
    "train_data_dir": "deepfashion_train_test_256/train_test_256/train",
    "test_data_dir": "deepfashion_train_test_256/train_test_256/test"
}

eval_params = {
    # "load_ckpt_path": "checkpoints/df_30.pt",
    "load_ckpt_path": "checkpoints/df.pt",
    "save_record_path": "dataset/test_dataset_record.json",
    "load_record_path": "dataset/test_dataset_record.json",
    "use_record": True, # use fix test dataset
    "save_dataset_record": False,
    "threshold" : 0.5,
    "positive_sample_rate": 0.5, # no implement
    "test_model_name": 'siamese', # TODO define at __init__
    "num_classes": 10393,
    "d_model": 2048,
    "device": 'cuda',
}