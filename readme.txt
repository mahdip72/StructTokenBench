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
proteinshake bindingsite continuous:

CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task
--config-name=proteinshake_binding_site.yaml   trainer.devices=1 trainer.fast_dev_run=false
  trainer.enable_progress_bar=true trainer.log_every_n_steps=1
  trainer.max_steps=60 trainer.limit_train_batches=0.2 trainer.limit_val_batches=1.0
  run_name="bindshake_continuous_$(date +%H%M%S)"   tokenizer=src.stb_tokenizers.WrappedMyRepBioLIP2Tokenizer
  tokenizer_device=cuda precompute_tokens=true   ++data.use_continuous=true
    ++tokenizer_kwargs.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_proteinshake.h5
    ++tokenizer_kwargs.embeddings_dataset=/   ++tokenizer_kwargs.force_reload=true
    ++tokenizer_kwargs.recompute=true   model.use_sequence=false data.use_sequence=false
    model.hidden_size=256 model.d_model=256   data.is_global_or_local=local data.target_field=binding_site
     data.data_path=/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/
      data.pdb_data_dir=/home/fe5vb/project/PST/pdb_data/
       ++lightning.callbacks.featureprobe._target_=src.script.debugging.FeatureProbe
        ++lightning.callbacks.h5check._target_=src.script.debugging.H5KeyCheck
         ++lightning.callbacks.h5check.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_proteinshake.h5
           ++lightning.callbacks.predstats._target_=src.script.debugging.DebugPredStatsCallback



*************************************************

interpro bindingsite:
CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task   --config-name=interpro.yaml
trainer.devices=1 trainer.fast_dev_run=false   trainer.enable_progress_bar=true
trainer.log_every_n_steps=1   trainer.max_steps=10000 trainer.limit_train_batches=0.2
trainer.limit_val_batches=1.0   run_name="interpro_binding_fixB_dim256_$(date +%H%M%S)"
tokenizer=src.stb_tokenizers.WrappedMyRepBioLIP2Tokenizer   tokenizer_device=cuda precompute_tokens=true
 ++data.use_continuous=true
 ++tokenizer_kwargs.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_interpro_binding_lite.h5
 ++tokenizer_kwargs.embeddings_dataset=/   ++tokenizer_kwargs.force_reload=true
 ++tokenizer_kwargs.recompute=true   model.use_sequence=false data.use_sequence=false
 model.d_model=128   data.is_global_or_local=local data.target_field=binding_label
 data.data_path=/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/
 data.pdb_data_dir=/home/fe5vb/project/PST/pdb_data/mmcif_files
 ++lightning.callbacks.featureprobe._target_=src.script.debugging.FeatureProbe
 ++lightning.callbacks.h5check._target_=src.script.debugging.H5KeyCheck
 ++lightning.callbacks.h5check.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_interpro_binding_lite.h5
 ++lightning.callbacks.predstats._target_=src.script.debugging.DebugPredStatsCallback

 ************************************************

 interpro activesite:
CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task
--config-name=interpro trainer.devices=1 trainer.fast_dev_run=false
trainer.enable_progress_bar=true trainer.log_every_n_steps=1 trainer.max_steps=10000
trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0
run_name="interpro_active_fix_$(date +%H%M%S)" tokenizer=src.stb_tokenizers.WrappedMyRepInterProTokenizer
tokenizer_device=cuda precompute_tokens=false ++data.use_continuous=true
++tokenizer_kwargs.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_interpro_activesite_lite.h5
++tokenizer_kwargs.embeddings_dataset=/ ++tokenizer_kwargs.fallback_to_any_chain=false
model.use_sequence=false data.use_sequence=false model.d_model=128
data.is_global_or_local=local data.target_field=activesite_label
data.data_path=/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/
data.pdb_data_dir=/home/fe5vb/project/PST/pdb_data/mmcif_files
data.data_name=src.dataset.interpro.InterProFunctionDataset

*********************************************************************

interpro conserved site:

CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task   --config-name=interpro
trainer.devices=1 trainer.fast_dev_run=false   trainer.enable_progress_bar=true trainer.log_every_n_steps=1
trainer.max_steps=10000 trainer.limit_train_batches=1.0 trainer.limit_val_batches=1.0
run_name="interpro_conserved_fix_$(date +%H%M%S)"   tokenizer=src.stb_tokenizers.WrappedMyRepInterProTokenizer
tokenizer_device=cuda precompute_tokens=false   ++data.use_continuous=true
++tokenizer_kwargs.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_conserved_lite.h5   ++tokenizer_kwargs.embeddings_dataset=/
++tokenizer_kwargs.fallback_to_any_chain=false   model.use_sequence=false data.use_sequence=false model.d_model=128
data.is_global_or_local=local data.target_field=conservedsite_label
data.data_path=/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/
data.pdb_data_dir=/home/fe5vb/project/PST/pdb_data/mmcif_files
data.data_name=src.dataset.interpro.InterProFunctionDataset

*********************************************************************
biolip2 catalyst:

 CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task 
 --config-name=biolip2.yaml trainer.devices=1 trainer.fast_dev_run=false 
 trainer.max_steps=10000 trainer.limit_train_batches=0.1 trainer.limit_val_batches=0.1 
 run_name="catbio_fixB_dim128_$(date +%H%M%S)" tokenizer=src.stb_tokenizers.WrappedMyRepBioLIP2Tokenizer 
 tokenizer_device=cuda precompute_tokens=true 
 +tokenizer_kwargs.h5_path="/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_biolip2_catalytic_lite.h5" 
 +tokenizer_kwargs.embeddings_dataset="/" +tokenizer_kwargs.force_reload=true 
 +tokenizer_kwargs.recompute=true model.use_sequence=false data.use_sequence=false 
 model.d_model=128 data.data_name=BioLIP2FunctionDataset data.is_global_or_local=local 
 data.target_field=catalytic_label 
 data.data_path="/home/fe5vb/project/PST/struct_token_bench_release_data/data/functional/local/" 
 data.pdb_data_dir="/home/fe5vb/project/PST/pdb_data/mmcif_files"

 ******************************************************************

 proteinglue:
 CUDA_VISIBLE_DEVICES=0 python -m src.script.run_supervised_task \
  --config-name=proteinglue_epitope_region \
  trainer.devices=1 \
  trainer.fast_dev_run=false \
  trainer.enable_progress_bar=true \
  trainer.log_every_n_steps=1 \
  trainer.max_steps=10000 \
  trainer.limit_train_batches=1.0 \
  trainer.limit_val_batches=1.0 \
  run_name="proteinglue_epitope_$(date +%H%M%S)" \
  tokenizer=src.stb_tokenizers.WrappedMyRepProteinGLUETokenizer \
  tokenizer_device=cuda \
  precompute_tokens=false \
  data.use_continuous=true \
  ++tokenizer_kwargs.h5_path=/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_proteinglue_lite.h5 \
  ++tokenizer_kwargs.embeddings_dataset=/ \
  ++tokenizer_kwargs.fallback_to_any_chain=false \
  model.use_sequence=false \
  data.use_sequence=false \
  model.d_model=128 \
  data.is_global_or_local=local \
  data.target_field=epitope_label \
  default_data_dir=/home/fe5vb/project/PST/struct_token_bench_release_data \
  data.pdb_data_dir=/home/fe5vb/project/PST/pdb_data

  ****************************************** 
  
flexRMSE:
python -m src.script.run_supervised_task --config-name atlas
default_data_dir="/home/fe5vb/project/PST/struct_token_bench_release_data"
deepspeed_path=null trainer.devices=1 run_name="atlas_flexRMSE"   tokenizer=WrappedMyRepAtlasTokenizer
tokenizer_device=cuda   precompute_tokens=false data.use_continuous=true data.use_sequence=false
model.use_sequence=false
model.d_model=128 data.target_field=rmsf_score
data.num_workers=16   +tokenizer_kwargs.h5_path="/mnt/hdd8/farzaneh/projects/PST/embeddings/vq_embed_atlas_lite.h5"
+tokenizer_kwargs.embeddings_dataset="/"
  lightning.callbacks.checkpoint.monitor=validation_spearmanr
  lightning.callbacks.checkpoint.mode=max
  lightning.callbacks.checkpoint.filename="\{epoch\}-\{step\}-\{validation_spearmanr:.4f\}"