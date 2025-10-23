biolip2 bindingsite:
CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task   --config-name=biolip2.yaml
trainer.devices=1 trainer.fast_dev_run=false   trainer.max_steps=10000 trainer.limit_train_batches=0.1
trainer.limit_val_batches=0.1   run_name="bindbio_fixB_dim128_$(date +%H%M%S)"   tokenizer=src.stb_tokenizers.WrappedMyRepBioLIP2Tokenizer
 tokenizer_device=cuda precompute_tokens=true   +tokenizer_kwargs.h5_path="/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_biolip2_binding_lite_model.h5"
   +tokenizer_kwargs.embeddings_dataset="/"   +tokenizer_kwargs.force_reload=true +tokenizer_kwargs.recompute=true
     model.use_sequence=false data.use_sequence=false   model.d_model=128
     data.data_name=BioLIP2FunctionDataset data.is_global_or_local=local data.target_field=binding_label
     data.data_path="/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/"
 data.pdb_data_dir="/home/fe5vb/project/PST/pdb_data/mmcif_files"
***********************************************





