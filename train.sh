######## LJSpeech ##########
# stage 1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 train_multi.py     --config config/LJSpeech/base_stage1.yaml --num_worker 16 

# stage 2 (improved)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 train_multi.py     --config config/LJSpeech/base_stage2_ict.yaml --num_worker 16 

# stage 3 (improved)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master_port=29500 train_multi_adv.py --config config/LJSpeech/base_stage3_ict.yaml --num_worker 16 


######## VCTK ##########
# stage 1
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 train_multi.py --config config/VCTK/base_stage1.yaml --num_worker 16 

# stage 2 (improved)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 train_multi.py --config config/VCTK/base_stage2_ict.yaml --num_worker 16 

# stage 3 (improved)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=29500 train_multi_adv.py --config config/VCTK/base_stage3_ict.yaml --num_worker 16 