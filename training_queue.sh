#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train.py --trait_name BFI_Conscientiousness --freeze --epochs 10 --checkpoint weights/BFI_Openness/best_model_epoch_14.pth
python train.py --trait_name BFI_Extraversion --freeze --epochs 10 --checkpoint weights/BFI_Openness/best_model_epoch_14.pth
python train.py --trait_name BFI_Agreeableness --freeze --epochs 10 --checkpoint weights/BFI_Openness/best_model_epoch_14.pth
python train.py --trait_name BFI_Neuroticism --freeze --epochs 10 --checkpoint weights/BFI_Openness/best_model_epoch_14.pth
python train.py --trait_name BPAQ_VerbalAggression --freeze --epochs 10 --checkpoint weights/BPAQ_Hostility/best_model_epoch_16.pth
python train.py --trait_name BPAQ_Anger --freeze --epochs 10 --checkpoint weights/BPAQ_Hostility/best_model_epoch_16.pth
python train.py --trait_name BPAQ_PhysicalAggression --freeze --epochs 10 --checkpoint weights/BPAQ_Hostility/best_model_epoch_16.pth
python train.py --trait_name DASS_Anxiety --freeze --epochs 10 --checkpoint weights/DASS_Depression/best_model_epoch_0.pth
python train.py --trait_name DASS_Stress --freeze --epochs 10 --checkpoint weights/DASS_Depression/best_model_epoch_0.pth
python train.py --trait_name OFER_AcuteFatigue --freeze --epochs 10 --checkpoint weights/OFER_ChronicFatigue/best_model_epoch_15.pth
python train.py --trait_name OFER_Recovery --freeze --epochs 10 --checkpoint weights/OFER_ChronicFatigue/best_model_epoch_15.pth