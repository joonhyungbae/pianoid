"""
Centralized model builder for all stages.

All inference/eval/precompute scripts should use these functions
instead of hardcoding model construction. When a new architecture is added,
only this file needs to be updated.
"""

import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# Stage 1: Wrist generation (MIDI → 6-dim wrist)
# ============================================================

def build_stage1_model(config: dict, device='cpu'):
    """Build Stage 1 wrist model (paper: BiLSTM h512 l6, v3 loss)."""
    from models.hand_baselines import HandLSTMRegression

    model_type = config.get('model_type', 'hand_bilstm')
    if model_type != 'hand_bilstm':
        raise ValueError(
            f"Unknown Stage 1 model_type: {model_type}. Supported: hand_bilstm"
        )

    hidden_dim = config.get('hidden_dim', config.get('latent_dim', 512))
    num_layers = config.get('num_layers', 6)
    use_dur_ioi = config.get('use_dur_ioi', False)
    midi_dim = 352 if use_dur_ioi else 264

    model = HandLSTMRegression(
        hand_dim=6, midi_dim=midi_dim,
        hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.1,
    )
    return model.to(device)


# ============================================================
# Stage 2: Torso generation (wrist → 12-dim torso)
# ============================================================

def build_stage2_model(config: dict, device='cpu'):
    """Build Stage 2 torso/backbone model from checkpoint config.

    Supported arch: transformer, lstm, mamba, film, mdm
    """
    from models.torso_baselines import (
        TorsoTransformerRegression, TorsoLSTMRegression,
        TorsoMambaRegression, TorsoFiLMRegression,
    )

    arch = config.get('arch', 'transformer')
    hidden_dim = config.get('latent_dim', config.get('hidden_dim', 256))
    num_layers = config.get('num_layers', 6)
    use_aria = config.get('use_aria', False)
    base_hand_dim = 12 if config.get('hand_velocity', False) else 6
    hand_dim = 0 if config.get('no_hand', False) else base_hand_dim
    output_dim = 30 if config.get('backbone', False) else 12

    temporal_cond_dim = 0
    if config.get('use_pitch_range', False):
        temporal_cond_dim += 1
    if config.get('use_onset_density', False):
        temporal_cond_dim += 1
    if config.get('use_onset_per_key', False):
        temporal_cond_dim += 88
    if config.get('use_piano_roll', False):
        temporal_cond_dim += 88

    if arch == 'mdm':
        from models.conditioned_mdm import ConditionedMDM
        model = ConditionedMDM(
            output_dim=output_dim,
            latent_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=max(1, hidden_dim // 64),
            dropout=0.1,
            cond_mask_prob=0.1,
            spatial_cond_dim=hand_dim,
            global_cond_dim=512 if use_aria else 0,
            temporal_cond_dim=temporal_cond_dim,
        )
    elif arch == 'transformer':
        model = TorsoTransformerRegression(
            hand_dim=hand_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=max(1, hidden_dim // 64),
            use_aria=use_aria, temporal_cond_dim=temporal_cond_dim,
        )
    elif arch == 'lstm':
        model = TorsoLSTMRegression(
            hand_dim=hand_dim, torso_dim=output_dim,
            hidden_dim=hidden_dim, num_layers=min(num_layers, 4),
            use_aria=use_aria, temporal_cond_dim=temporal_cond_dim,
        )
    elif arch == 'mamba':
        model = TorsoMambaRegression(
            hand_dim=hand_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            use_aria=use_aria, temporal_cond_dim=temporal_cond_dim,
        )
    elif arch == 'film':
        model = TorsoFiLMRegression(
            hand_dim=hand_dim, hidden_dim=hidden_dim, num_layers=num_layers,
            use_aria=use_aria, temporal_cond_dim=temporal_cond_dim,
        )
    else:
        raise ValueError(
            f"Unknown Stage 2 arch: {arch}. "
            f"Supported: transformer, lstm, mamba, film, mdm"
        )

    return model.to(device)


# ============================================================
# Stage 3: Arm generation (wrist+chest → 12-dim arms)
# ============================================================

def build_stage3_arm_model(config: dict, device='cpu'):
    """Build Stage 3 arm model from checkpoint config.

    Supported arch: mdm, transformer, bilstm, film
    """
    arch = config.get('arch', 'mdm')
    hidden_dim = config.get('latent_dim', config.get('hidden_dim', 256))
    num_layers = config.get('num_layers', 6)
    use_aria = config.get('use_aria', True)
    spatial_dim = config.get('spatial_dim', 18)
    arm_dim = config.get('arm_dim', 6)  # elbows only (2 joints x 3)

    temporal_cond_dim = 0
    if config.get('use_onset_density', False):
        temporal_cond_dim += 1
    if config.get('use_pitch_range', False):
        temporal_cond_dim += 1
    if config.get('use_piano_roll', False):
        temporal_cond_dim += 88

    if arch == 'mdm':
        from models.hierarchical_mdm import ConditionedMDM
        model = ConditionedMDM(
            output_dim=arm_dim,
            latent_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config.get('num_heads', 4),
            spatial_cond_dim=spatial_dim,
            global_cond_dim=512 if use_aria else 0,
            temporal_cond_dim=temporal_cond_dim,
        )
    elif arch in ('transformer', 'bilstm', 'mamba', 'film'):
        from models.arm_baselines import (
            ArmTransformerRegression, ArmBiLSTMRegression,
            ArmMambaRegression, ArmFiLMRegression,
        )
        cls_map = {
            'transformer': ArmTransformerRegression,
            'bilstm': ArmBiLSTMRegression,
            'mamba': ArmMambaRegression,
            'film': ArmFiLMRegression,
        }
        model = cls_map[arch](
            spatial_dim=spatial_dim,
            arm_dim=arm_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_aria=use_aria,
            temporal_cond_dim=temporal_cond_dim,
        )
    else:
        raise ValueError(
            f"Unknown Stage 3 arch: {arch}. "
            f"Supported: mdm, transformer, bilstm, film"
        )

    return model.to(device)


def build_stage3_head_model(config: dict, device='cpu'):
    """Build Stage 3 HeadRegressor from checkpoint config."""
    # Simple MLP: C7+CLAV (6-dim) → 4 head joints (12-dim)
    hidden_dim = config.get('hidden_dim', 256)
    num_layers = config.get('num_layers', 3)

    layers = [nn.Linear(6, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, 12))
    model = nn.Sequential(*layers)

    return model.to(device)


# ============================================================
# Stage 4: Lower body generation (spine → 30-dim lower)
# ============================================================

def build_stage4_model(config: dict, device='cpu'):
    """Build Stage 4 lower body model from checkpoint config."""
    try:
        from models.lower_body import HandConditionedLowerGenerator
        model = HandConditionedLowerGenerator(
            hand_dim=config.get('hand_dim', 6),
            hidden_dim=config.get('hidden_dim', config.get('latent_dim', 256)),
            num_layers=config.get('num_layers', 4),
            use_aria=config.get('use_aria', False),
        )
    except ImportError:
        raise ImportError("Could not import lower body model from models.lower_body")

    return model.to(device)


# ============================================================
# Convenience: load checkpoint and build model in one call
# ============================================================

def load_stage1(ckpt_path: str, device='cpu'):
    """Load Stage 1 checkpoint → (model, config)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    model = build_stage1_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def load_stage2(ckpt_path: str, device='cpu'):
    """Load Stage 2 checkpoint → (model, config)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model = build_stage2_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def load_stage3_arm(ckpt_path: str, device='cpu'):
    """Load Stage 3 arm checkpoint → (model, config)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model = build_stage3_arm_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def load_stage4(ckpt_path: str, device='cpu'):
    """Load Stage 4 checkpoint → (model, config)."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    model = build_stage4_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config
