
## LJSpeech 
python test.py --weight_path ./checkpoints/RapFlow-TTS-LJS-Stage3-Improved --model_name RapFlow-TTS-LJS-Stage3-Improved --n_timesteps 2  --weight_name model-last --device cuda:0

## VCTK
python test.py --weight_path ./checkpoints/RapFlow-TTS-VCTK-Stage3-Improved --model_name RapFlow-TTS-VCTK-Stage3-Improved --n_timesteps 2  --weight_name model-last --device cuda:0

