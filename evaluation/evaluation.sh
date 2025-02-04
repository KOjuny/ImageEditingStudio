#!/bin/bash

# GPU 0번을 사용하도록 설정
export CUDA_VISIBLE_DEVICES=7

# Python 스크립트 실행

python evaluate.py \
    --annotation_mapping_file "/home/poong/junseok/PIE_Bench/mapping_file.json" \
    --metrics structure_distance psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part \
    --src_image_folder "/home/poong/junseok/PIE_Bench/annotation_images" \
    --tgt_methods 1_ddim+masactrl 1_null-text-inversion+masactrl 1_directinversion+masactrl \
    --result_path masactrl_results.csv \
    --evaluate_whole_table
