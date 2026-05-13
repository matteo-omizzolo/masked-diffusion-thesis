"""Tiny HollowMD4/Text8 config for Stage 1 training-loop smoke.

This is intentionally not a paper-quality config. It keeps the HollowMD4 model
family but shrinks the network and step count so the training loop, checkpoint
save, eval path, and resume path can be tested safely.
"""

from __future__ import annotations

import os

from ml_collections import config_dict


def get_config() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()

    config.vocab_size = 27
    config.dataset = "text8"
    config.classes = -1
    config.task_type = "text"
    config.model_type = "hollow_md4"
    config.data_shape = (256,)

    config.timesteps = 100
    config.noise_schedule = "linear"
    config.outside_embed = True
    config.time_features = "t"
    config.cont_time = True

    config.feature_dim = 32
    config.hidden_dim = 128
    config.n_layers = 2
    config.n_layers_per_mixed = 1
    config.ch_mult = (1,)
    config.n_dit_layers = 0
    config.dit_num_heads = 4
    config.dit_hidden_size = 128
    config.dropout_rate = 0.0
    config.num_heads = 4
    config.mlp_type = "glu"
    config.depth_scaled_init = True
    config.cond_type = "adaln_zero"

    config.learning_rate = 1e-4
    config.learning_rate_schedule = "constant"
    config.warmup_steps = 1
    config.weight_decay = 0.0
    config.clip = 0.0
    config.b2 = 0.999
    config.num_epochs = -1
    config.ema_rate = 0.0
    config.num_train_steps = int(os.environ.get("IC_STAGE1_STEPS", "50"))
    config.num_eval_steps = 1
    config.batch_size = int(os.environ.get("IC_STAGE1_BATCH_SIZE", "8"))
    config.num_microbatches = 1
    config.per_device_batch_size = -1
    config.eval_pad_last_batch = False
    config.check_nans = True

    config.sampler = "gibbs"
    config.sampling_grid = "cosine"
    config.topp = 0.98
    config.k = 1
    config.gibbs_temp = 1.0

    config.log_loss_every_steps = 5
    config.eval_every_steps = 1_000_000
    config.checkpoint_every_steps = 25
    config.checkpoint_keep_period = 25
    config.seed = 0
    config.grain_num_workers = 0
    config.trial = 0
    config.test_in_colab = False

    config.wandbentity = os.environ.get("WANDB_ENTITY", "")
    config.wandbname = os.environ.get("WANDB_NAME", "ic_text8_stage1_tiny")
    config.vocab_dir = os.environ.get(
        "TEXT8_WORD_VOCAB_PKL", "data/text8_md4/text8_word_vocab.pkl"
    )
    config.loss_type = "masked"

    return config

