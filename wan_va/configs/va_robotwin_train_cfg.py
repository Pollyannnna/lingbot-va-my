# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'

default_dataset_path = "/data/share/lijiang/data/robotwin-lingbotva/lerobot_robotwin_eef_clean_50"
va_robotwin_train_cfg.dataset_path = os.getenv(
    "ROBOTWIN_DATASET_PATH",
    default_dataset_path,
)
va_robotwin_train_cfg.empty_emb_path = os.getenv(
    "ROBOTWIN_EMPTY_EMB_PATH",
    os.path.join(va_robotwin_train_cfg.dataset_path, "empty_emb.pt"),
)

resume_from = os.getenv("ROBOTWIN_RESUME_FROM", "").strip()
if resume_from:
    va_robotwin_train_cfg.resume_from = resume_from

va_robotwin_train_cfg.enable_swanlab = os.getenv(
    "ROBOTWIN_ENABLE_SWANLAB", os.getenv("ROBOTWIN_ENABLE_WANDB", "1")
) != "0"
va_robotwin_train_cfg.enable_wandb = va_robotwin_train_cfg.enable_swanlab
va_robotwin_train_cfg.load_worker = int(os.getenv("ROBOTWIN_LOAD_WORKER", "16"))
va_robotwin_train_cfg.dataset_init_worker = int(
    os.getenv("ROBOTWIN_DATASET_INIT_WORKER", "8")
)
va_robotwin_train_cfg.prefetch_factor = int(
    os.getenv("ROBOTWIN_DATALOADER_PREFETCH", "2")
)
va_robotwin_train_cfg.pin_memory = os.getenv("ROBOTWIN_PIN_MEMORY", "1") != "0"
va_robotwin_train_cfg.persistent_workers = (
    os.getenv("ROBOTWIN_PERSISTENT_WORKERS", "1") != "0"
)
va_robotwin_train_cfg.save_interval = int(os.getenv("ROBOTWIN_SAVE_INTERVAL", "1000"))
va_robotwin_train_cfg.gc_interval = int(os.getenv("ROBOTWIN_GC_INTERVAL", "50"))
va_robotwin_train_cfg.cfg_prob = float(os.getenv("ROBOTWIN_CFG_PROB", "0.1"))

# Training parameters
va_robotwin_train_cfg.learning_rate = float(os.getenv("ROBOTWIN_LR", "1e-5"))
va_robotwin_train_cfg.beta1 = float(os.getenv("ROBOTWIN_BETA1", "0.9"))
va_robotwin_train_cfg.beta2 = float(os.getenv("ROBOTWIN_BETA2", "0.95"))
va_robotwin_train_cfg.weight_decay = float(os.getenv("ROBOTWIN_WEIGHT_DECAY", "0.1"))
va_robotwin_train_cfg.warmup_steps = int(os.getenv("ROBOTWIN_WARMUP_STEPS", "10"))
va_robotwin_train_cfg.batch_size = int(os.getenv("ROBOTWIN_BATCH_SIZE", "1"))
va_robotwin_train_cfg.gradient_accumulation_steps = int(
    os.getenv("ROBOTWIN_GRAD_ACC", "1")
)
va_robotwin_train_cfg.num_steps = int(os.getenv("ROBOTWIN_NUM_STEPS", "50000"))
