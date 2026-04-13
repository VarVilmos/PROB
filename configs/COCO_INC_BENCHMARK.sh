#!/usr/bin/env bash
# COCO_INC: Class-Incremental Learning on COCO — 8 tasks, 10 classes each
#
# Tasks:
#   T1: classes  0- 9  (aeroplane … cow)
#   T2: classes 10-19  (diningtable … tvmonitor)
#   T3: classes 20-29  (truck … giraffe)
#   T4: classes 30-39  (backpack … refrigerator)
#   T5: classes 40-49  (frisbee … tennis racket)
#   T6: classes 50-59  (banana … cake)
#   T7: classes 60-69  (bed … vase)
#   T8: classes 70-79  (scissors … bowl)
#
# Usage (single GPU, Colab):
#   bash configs/COCO_INC_BENCHMARK.sh
#
# Usage (multi-GPU, 4 GPUs):
#   GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 configs/COCO_INC_BENCHMARK.sh
#
# To run classifier-only (freeze detector, train only class_embed):
#   add --freeze_detector to every task below
#
# Before running: generate split files once with:
#   python datasets/generate_coco_inc_splits.py

set -x

EXP_DIR=exps/COCO_INC/PROB
WANDB_NAME=COCO_INC_PROB
PY_ARGS=${@:1}

# ── Task 1: classes 0-9 ────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t1" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t1_train' --test_set 'coco_inc_all_test' \
    --epochs 41 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t1" \
    --exemplar_replay_selection --exemplar_replay_max_length 425 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_cur_file "learned_coco_inc_t1_ft.txt" \
    ${PY_ARGS}


# ── Task 2: classes 10-19 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 10 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t2_train' --test_set 'coco_inc_all_test' \
    --epochs 51 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t2" \
    --exemplar_replay_selection --exemplar_replay_max_length 870 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t1_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t2_ft.txt" \
    --pretrain "${EXP_DIR}/t1/checkpoint0040.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 2 fine-tune (on replay exemplars) ─────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t2_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 10 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t2_ft" --test_set 'coco_inc_all_test' \
    --epochs 111 --lr_drop 40 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t2_ft" \
    --pretrain "${EXP_DIR}/t2/checkpoint0050.pth" \
    ${PY_ARGS}


# ── Task 3: classes 20-29 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t3_train' --test_set 'coco_inc_all_test' \
    --epochs 121 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t3" \
    --exemplar_replay_selection --exemplar_replay_max_length 1180 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t2_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t3_ft.txt" \
    --pretrain "${EXP_DIR}/t2_ft/checkpoint0110.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 3 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t3_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 20 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t3_ft" --test_set 'coco_inc_all_test' \
    --epochs 181 --lr_drop 35 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t3_ft" \
    --pretrain "${EXP_DIR}/t3/checkpoint0120.pth" \
    ${PY_ARGS}


# ── Task 4: classes 30-39 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 30 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t4_train' --test_set 'coco_inc_all_test' \
    --epochs 191 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t4" \
    --exemplar_replay_selection --exemplar_replay_max_length 1370 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t3_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t4_ft.txt" \
    --pretrain "${EXP_DIR}/t3_ft/checkpoint0180.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 4 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t4_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 30 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t4_ft" --test_set 'coco_inc_all_test' \
    --epochs 261 --lr_drop 50 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t4_ft" \
    --pretrain "${EXP_DIR}/t4/checkpoint0190.pth" \
    ${PY_ARGS}


# ── Task 5: classes 40-49 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t5" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t5_train' --test_set 'coco_inc_all_test' \
    --epochs 271 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t5" \
    --exemplar_replay_selection --exemplar_replay_max_length 1500 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t4_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t5_ft.txt" \
    --pretrain "${EXP_DIR}/t4_ft/checkpoint0260.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 5 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t5_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t5_ft" --test_set 'coco_inc_all_test' \
    --epochs 341 --lr_drop 35 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t5_ft" \
    --pretrain "${EXP_DIR}/t5/checkpoint0270.pth" \
    ${PY_ARGS}


# ── Task 6: classes 50-59 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t6" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 50 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t6_train' --test_set 'coco_inc_all_test' \
    --epochs 351 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t6" \
    --exemplar_replay_selection --exemplar_replay_max_length 1600 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t5_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t6_ft.txt" \
    --pretrain "${EXP_DIR}/t5_ft/checkpoint0340.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 6 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t6_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 50 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t6_ft" --test_set 'coco_inc_all_test' \
    --epochs 421 --lr_drop 35 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t6_ft" \
    --pretrain "${EXP_DIR}/t6/checkpoint0350.pth" \
    ${PY_ARGS}


# ── Task 7: classes 60-69 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t7" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t7_train' --test_set 'coco_inc_all_test' \
    --epochs 431 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t7" \
    --exemplar_replay_selection --exemplar_replay_max_length 1750 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t6_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t7_ft.txt" \
    --pretrain "${EXP_DIR}/t6_ft/checkpoint0420.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 7 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t7_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 60 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t7_ft" --test_set 'coco_inc_all_test' \
    --epochs 501 --lr_drop 35 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t7_ft" \
    --pretrain "${EXP_DIR}/t7/checkpoint0430.pth" \
    ${PY_ARGS}


# ── Task 8: classes 70-79 ──────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t8" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 70 --CUR_INTRODUCED_CLS 10 \
    --train_set 'coco_inc_t8_train' --test_set 'coco_inc_all_test' \
    --epochs 511 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 --freeze_prob_model \
    --wandb_name "${WANDB_NAME}_t8" \
    --exemplar_replay_selection --exemplar_replay_max_length 1875 \
    --exemplar_replay_dir ${WANDB_NAME} \
    --exemplar_replay_prev_file "learned_coco_inc_t7_ft.txt" \
    --exemplar_replay_cur_file "learned_coco_inc_t8_ft.txt" \
    --pretrain "${EXP_DIR}/t7_ft/checkpoint0500.pth" --lr 2e-5 \
    ${PY_ARGS}


# ── Task 8 fine-tune ───────────────────────────────────────────────────────
python -u main_open_world.py \
    --output_dir "${EXP_DIR}/t8_ft" \
    --dataset COCO_INC --PREV_INTRODUCED_CLS 70 --CUR_INTRODUCED_CLS 10 \
    --train_set "${WANDB_NAME}/learned_coco_inc_t8_ft" --test_set 'coco_inc_all_test' \
    --epochs 581 --lr_drop 35 \
    --model_type 'prob' --obj_loss_coef 8e-4 --obj_temp 1.3 \
    --wandb_name "${WANDB_NAME}_t8_ft" \
    --pretrain "${EXP_DIR}/t8/checkpoint0510.pth" \
    ${PY_ARGS}
