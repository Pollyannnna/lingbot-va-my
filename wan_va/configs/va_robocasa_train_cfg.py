# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict

from .va_robocasa_cfg import va_robocasa_cfg


va_robocasa_train_cfg = EasyDict(__name__="Config: VA robocasa train")
va_robocasa_train_cfg.update(va_robocasa_cfg)

default_dataset_path = "/data/share/lijiang/data/robocasa365/target-human-50"
va_robocasa_train_cfg.dataset_path = os.getenv(
    "ROBOCASA_DATASET_PATH",
    default_dataset_path,
)
va_robocasa_train_cfg.empty_emb_path = os.getenv(
    "ROBOCASA_EMPTY_EMB_PATH",
    os.path.join(va_robocasa_train_cfg.dataset_path, "empty_emb.pt"),
)

resume_from = os.getenv("ROBOCASA_RESUME_FROM", "").strip()
if resume_from:
    va_robocasa_train_cfg.resume_from = resume_from

va_robocasa_train_cfg.enable_swanlab = os.getenv(
    "ROBOCASA_ENABLE_SWANLAB", os.getenv("ROBOCASA_ENABLE_WANDB", "1")
) != "0"
va_robocasa_train_cfg.enable_wandb = va_robocasa_train_cfg.enable_swanlab
va_robocasa_train_cfg.load_worker = int(os.getenv("ROBOCASA_LOAD_WORKER", "16"))
va_robocasa_train_cfg.dataset_init_worker = int(
    os.getenv("ROBOCASA_DATASET_INIT_WORKER", "8")
)
va_robocasa_train_cfg.prefetch_factor = int(
    os.getenv("ROBOCASA_DATALOADER_PREFETCH", "2")
)
va_robocasa_train_cfg.pin_memory = os.getenv("ROBOCASA_PIN_MEMORY", "1") != "0"
va_robocasa_train_cfg.persistent_workers = (
    os.getenv("ROBOCASA_PERSISTENT_WORKERS", "1") != "0"
)
va_robocasa_train_cfg.save_interval = int(os.getenv("ROBOCASA_SAVE_INTERVAL", "1000"))
va_robocasa_train_cfg.gc_interval = int(os.getenv("ROBOCASA_GC_INTERVAL", "50"))
va_robocasa_train_cfg.cfg_prob = float(os.getenv("ROBOCASA_CFG_PROB", "0.1"))

# Training parameters
va_robocasa_train_cfg.learning_rate = float(os.getenv("ROBOCASA_LR", "1e-5"))
va_robocasa_train_cfg.beta1 = float(os.getenv("ROBOCASA_BETA1", "0.9"))
va_robocasa_train_cfg.beta2 = float(os.getenv("ROBOCASA_BETA2", "0.95"))
va_robocasa_train_cfg.weight_decay = float(os.getenv("ROBOCASA_WEIGHT_DECAY", "0.1"))
va_robocasa_train_cfg.warmup_steps = int(os.getenv("ROBOCASA_WARMUP_STEPS", "10"))
va_robocasa_train_cfg.batch_size = int(os.getenv("ROBOCASA_BATCH_SIZE", "1"))
va_robocasa_train_cfg.gradient_accumulation_steps = int(
    os.getenv("ROBOCASA_GRAD_ACC", "1")
)
va_robocasa_train_cfg.num_steps = int(os.getenv("ROBOCASA_NUM_STEPS", "50000"))
