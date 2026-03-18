# RoboCasa Client for LingBot-VA

This folder adds a RoboCasa benchmark client that reuses LingBot-VA's websocket inference server protocol.

## Files

- `eval_policy_client.py`: RoboCasa evaluation client.
- `launch_server.sh`: helper to launch LingBot-VA server.
- `launch_client.sh`: helper to run RoboCasa evaluation.

## 1) Start server

```bash
cd lingbot-va
CONFIG_NAME=robocasa START_PORT=29056 bash evaluation/robocasa/launch_server.sh
```

If you have a RoboCasa-specific LingBot-VA server config, replace `CONFIG_NAME` accordingly.

## 2) Run benchmark client

Single task:

```bash
cd lingbot-va
bash evaluation/robocasa/launch_client.sh TurnOffMicrowave 50 results/robocasa
```

All 24 RoboCasa tasks:

```bash
python -m evaluation.robocasa.eval_policy_client \
  --host 127.0.0.1 \
  --port 29056 \
  --task-name all \
  --num-trials-per-task 50 \
  --save-dir results/robocasa
```

## Notes

- The client follows the RoboCasa setup from `cosmos-policy`:
  - scene cycling via `layout_and_style_ids`
  - task success via `env._check_success()`
  - task horizon from RoboCasa task defaults
- Default controller config file path points to:
  - `other/cosmos-policy/cosmos_policy/experiments/robot/robocasa/robocasa_controller_configs.pkl`
- The default client now reconstructs PandaOmron env actions using the RoboCasa / robosuite runtime order:
  - `ee_position(3), ee_rotation(3), gripper_close(1), base_motion(4), control_mode(1)`
- If your server returns non-12D actions, `--action-map-mode auto` will map the first 7 model dims to the runtime manipulator action `ee_position + ee_rotation + gripper_close`, then append fixed `base_motion/control_mode`.
- Default fixed prefix is `base_motion=[0,0,0,0]`, `control_mode=-1`. Override with `--fixed-base-motion` / `--fixed-control-mode` if your env/controller expects different values.

## Post-training on RoboCasa

Prepare data and stats:

```bash
cd lingbot-va
DATASET_ROOT=/path/to/robocasa_dataset_root \
MODEL_ROOT=/path/to/model_root \
bash script/prepare_robocasa_for_va.sh
```

Run training:

```bash
cd lingbot-va
export ROBOCASA_DATASET_PATH=/path/to/robocasa_dataset_root
export ROBOCASA_EMPTY_EMB_PATH=${ROBOCASA_DATASET_PATH}/empty_emb.pt
export ROBOCASA_NORM_STATS_PATH=${ROBOCASA_DATASET_PATH}/robocasa_action_stats_for_lingbotva.json
NGPU=8 bash script/run_va_posttrain_robocasa.sh
```
