#!/bin/bash
# launch_client_with_logging.sh
# Drop-in replacement for launch_client.sh that uses eval_with_logging.py
# for detailed per-episode inference logging.
#
# Usage: bash evaluation/robotwin/launch_client_with_logging.sh [save_root] [task_name]

export LD_LIBRARY_PATH=/usr/lib64:/usr/lib:$LD_LIBRARY_PATH

# Fix: ensure glvnd dirs exist (needed by sapien at import)
mkdir -p /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d

task_groups=(
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

save_root=${1:-'./results'}
task_name=${2:-"adjust_bottle"}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
PORT=29056

# Use eval_with_logging instead of eval_polict_client_openpi
PYTHONWARNINGS=ignore::UserWarning \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python -m evaluation.robotwin.eval_with_logging --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --save_root ${save_root} \
    --video_guidance_scale 5 \
    --action_guidance_scale 1 \
    --test_num 100 \
    --port ${PORT}
