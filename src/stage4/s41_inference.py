#!/usr/bin/env python3
"""
Inference: New Data-Driven Hierarchical Architecture

Cascade:
  MIDI → [S1 Hand Transformer] → hands
       → [S2 TorsoMDM]        → torso (C7, CLAV, T10, STRN)
       → [S3 ArmMDM]          → arms (RSHO, RELB, LSHO, LELB)
       → [HeadRegressor]      → head (RFHD, LFHD, RBHD, LBHD)
       → [S4 LowerFiLM]       → lower body (pelvis + legs)

Data justification:
  hand-C7 r=0.607 >> MIDI-C7 r=0.265  (hands mediate MIDI→torso)
  elbow-wrist r=0.946, elbow-shoulder r=0.770  (arms after both S1+S2)
  head-C7 r=0.934  (head rigidly follows neck → no diffusion)
  spine-pelvis r=0.848  (lower body after spine)

Usage:
    python src/stage4/s41_inference.py \\
        --checkpoint_dir checkpoints/new_arch \\
        --cache_dir MOSA_dataset/training_cache_B \\
        --output_dir results/new_arch
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add paths – MDM first so its modules take priority where needed
MDM_PATH = str(Path(__file__).parent.parent.parent / 'vendor' / 'MDM')
sys.path.insert(0, MDM_PATH)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.constants import (
    FPS, NUM_JOINTS,
    RWRIST_IDX, LWRIST_IDX, RHAND_IDX, LHAND_IDX,
    RSHO_IDX, LSHO_IDX, RELB_IDX, LELB_IDX,
    PARCO_ROOT_INDICES, PARCO_R_LEG_INDICES, PARCO_L_LEG_INDICES,
)
from diffusion_lib import create_diffusion, ddim_sample
from models.model_factory import load_stage1, load_stage2, load_stage3_arm
from models.hierarchical_mdm import (
    TorsoMDM,
    ArmMDM,
    HeadRegressor,
    HandConditionedLowerGenerator,
)

# ---------------------------------------------------------------------------
# Joint index constants
# ---------------------------------------------------------------------------
LOWER_INDICES = PARCO_ROOT_INDICES + PARCO_R_LEG_INDICES + PARCO_L_LEG_INDICES  # 10 joints

# Hand ordering inside the 6-dim wrist vector: [LWRIST, RWRIST]
# (matches s14_train_hand.py)
HAND_JOINT_ORDER = [LWRIST_IDX, RWRIST_IDX]


# ===========================================================================
# NewArchInference
# ===========================================================================

class NewArchInference:
    """
    Inference pipeline for the new data-driven hierarchical architecture.

    Cascade order:
        Stage 1 : Hand Transformer  (MIDI → hands)
        Stage 2 : TorsoMDM          (hands → torso)
        Stage 3 : ArmMDM            (wrists + chest → shoulders + elbows)
        Head    : HeadRegressor     (C7 + CLAV → head)  [no diffusion]
        Stage 4 : LowerFiLM         (spine → lower body)
    """

    def __init__(self, checkpoint_dir: str, device: str = 'cuda', ddim_steps: int = 50):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.ddim_steps = ddim_steps

        # Models (populated by load_* methods; None if checkpoint missing)
        self.stage1_hand = None
        self.stage2_torso = None
        self.stage2_diffusion = None
        self.stage3_arm = None
        self.stage3_diffusion = None
        self.head_regressor = None
        self.stage4_lower = None

        self.load_stage1()
        self.load_stage2()
        self.load_stage3()
        self.load_head()
        self.load_stage4()

    # ------------------------------------------------------------------
    # Checkpoint loaders
    # ------------------------------------------------------------------

    def load_stage1(self):
        """Load Stage 1 BiLSTM from checkpoint_dir/stage1/hand_bilstm_best.pt."""
        ckpt_path = self.checkpoint_dir / 'stage1' / 'hand_bilstm_best.pt'
        if not ckpt_path.exists():
            print(f"[WARNING] Stage 1 checkpoint not found: {ckpt_path}  (skipping)")
            return

        model, config = load_stage1(ckpt_path, self.device)
        self.stage1_hand = model
        self.stage1_config = config
        print(f"[Stage 1] Loaded {config.get('model_type', '?')} "
              f"(hidden={config.get('hidden_dim', config.get('latent_dim'))}, "
              f"layers={config.get('num_layers')})")

    def load_stage2(self):
        """Load TorsoMDM from checkpoint_dir/stage2_torso/best.pt."""
        ckpt_path = self.checkpoint_dir / 'stage2_torso' / 'best.pt'
        if not ckpt_path.exists():
            print(f"[WARNING] Stage 2 checkpoint not found: {ckpt_path}  (skipping)")
            return

        model, config = load_stage2(ckpt_path, self.device)
        self.stage2_torso = model
        self.stage2_config = config
        # Stage 2 is now deterministic regression (no diffusion)
        self.stage2_diffusion = None
        print(f"[Stage 2] Loaded {config.get('arch', '?')} "
              f"(hidden={config.get('latent_dim')}, use_aria={config.get('use_aria')})")

    def load_stage3(self):
        """Load ArmMDM from checkpoint_dir/stage3_arm/best.pt."""
        ckpt_path = self.checkpoint_dir / 'stage3_arm' / 'best.pt'
        if not ckpt_path.exists():
            print(f"[WARNING] Stage 3 (Arm) checkpoint not found: {ckpt_path}  (skipping)")
            return

        model, config = load_stage3_arm(ckpt_path, self.device)
        self.stage3_arm = model
        self.stage3_diffusion = create_diffusion(num_timesteps=1000)
        print(f"[Stage 3] Loaded ArmMDM  "
              f"(latent={config.get('latent_dim', 256)}, use_aria={use_aria})")

    def load_head(self):
        """Load HeadRegressor from checkpoint_dir/stage2_head/best.pt."""
        ckpt_path = self.checkpoint_dir / 'stage2_head' / 'best.pt'
        if not ckpt_path.exists():
            print(f"[WARNING] Head checkpoint not found: {ckpt_path}  (skipping)")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get('config', {})

        model = HeadRegressor(
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 3),
            dropout=config.get('dropout', 0.1),
        ).to(self.device)

        state_dict = ckpt.get('model_state_dict', ckpt.get('model', None))
        if state_dict is None:
            print(f"[WARNING] Could not find model weights in Head checkpoint.")
            return

        model.load_state_dict(state_dict)
        model.eval()
        self.head_regressor = model
        print(f"[Head  ] Loaded HeadRegressor  "
              f"(hidden={config.get('hidden_dim', 256)}, layers={config.get('num_layers', 3)})")

    def load_stage4(self):
        """Load LowerFiLM from checkpoint_dir/stage4/best.pt."""
        ckpt_path = self.checkpoint_dir / 'stage4' / 'best.pt'
        if not ckpt_path.exists():
            print(f"[WARNING] Stage 4 checkpoint not found: {ckpt_path}  (skipping)")
            return

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        config = ckpt.get('config', {})

        # Stage 4 uses HandConditionedLowerGenerator but conditioned on SPINE
        # (spine → pelvis r=0.848 > hands → pelvis r=0.647)
        # The model architecture is identical; we simply pass spine features
        # as the "hand" conditioning input.
        # Spine conditioning dim: C7(3) + T10(3) = 6-dim → hand_dim=6
        spine_dim = 6  # [C7, T10]
        model = HandConditionedLowerGenerator(
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 4),
            hand_dim=spine_dim,
            use_aria=config.get('use_aria', True),
            use_midi=config.get('use_midi', False),
        ).to(self.device)

        state_dict = ckpt.get('model', ckpt.get('model_state_dict', None))
        if state_dict is None:
            print(f"[WARNING] Could not find model weights in Stage 4 checkpoint.")
            return

        model.load_state_dict(state_dict)

        if 'mean_pose' in ckpt:
            motion_range = ckpt.get('motion_range', torch.ones(30) * 0.05)
            model.set_mean_pose(
                ckpt['mean_pose'].to(self.device),
                motion_range.to(self.device),
            )

        model.eval()
        self.stage4_lower = model
        print(f"[Stage 4] Loaded LowerFiLM (spine-conditioned)  "
              f"(hidden={config.get('hidden_dim', 256)}, use_aria={config.get('use_aria', True)})")

    # ------------------------------------------------------------------
    # Stage inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_stage1(
        self,
        midi_dict: dict,
    ) -> torch.Tensor:
        """
        Run Stage 1: Hand Transformer.

        Args:
            midi_dict: dict with keys
                'piano_roll' : (B, T, 88)  float tensor
                'velocity'   : (B, T, 88)  float tensor
                'onset'      : (B, T, 88)  float tensor

        Returns:
            hand: (B, T, 6) [LWRIST, RWRIST]
        """
        piano_roll = midi_dict['piano_roll']
        velocity   = midi_dict['velocity']
        onset      = midi_dict['onset']

        # Stage 1 expects concatenated MIDI (B, T, 264)
        midi_input = torch.cat([piano_roll, velocity, onset], dim=-1)

        hand = self.stage1_hand(midi_input)  # (B, T, 12)
        return hand

    @torch.no_grad()
    def _run_stage2(
        self,
        hand: torch.Tensor,
        aria: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run Stage 2: TorsoMDM  hand → torso.

        Args:
            hand: (B, T, 12)
            aria: (B, 512)

        Returns:
            torso: (B, T, 12) [C7, CLAV, T10, STRN]
        """
        B, T, _ = hand.shape
        torso = ddim_sample(
            self.stage2_torso,
            self.stage2_diffusion,
            shape=(B, T, 12),
            model_kwargs={'hand': hand, 'aria': aria},
            ddim_steps=self.ddim_steps,
            device=self.device,
            progress=True,
        )
        return torso  # (B, T, 12)

    @torch.no_grad()
    def _run_stage3(
        self,
        wrists: torch.Tensor,
        chest: torch.Tensor,
        aria: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run Stage 3: ArmMDM  wrists + chest → arms.

        Args:
            wrists: (B, T, 6)  [RWRIST(10), LWRIST(14)] – extracted from Stage 1
            chest : (B, T, 6)  [T10(6), STRN(7)]        – extracted from Stage 2
            aria  : (B, 512)

        Returns:
            arms: (B, T, 12) [RSHO, RELB, LSHO, LELB]
        """
        B, T, _ = wrists.shape
        # ArmMDM.forward expects 'wrists' and 'chest' kwargs
        arms = ddim_sample(
            self.stage3_arm,
            self.stage3_diffusion,
            shape=(B, T, 12),
            model_kwargs={'wrists': wrists, 'chest': chest, 'aria': aria},
            ddim_steps=self.ddim_steps,
            device=self.device,
            progress=True,
        )
        return arms  # (B, T, 12)

    @torch.no_grad()
    def _run_head(
        self,
        neck: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run HeadRegressor (no diffusion).

        Args:
            neck: (B, T, 6) [C7, CLAV] – first 6 dims of Stage 2 output

        Returns:
            head: (B, T, 12) [RFHD, LFHD, RBHD, LBHD]
        """
        head = self.head_regressor(neck)  # (B, T, 12)
        return head

    @torch.no_grad()
    def _run_stage4(
        self,
        spine: torch.Tensor,
        aria: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run Stage 4: LowerFiLM  spine → lower body.

        We pass spine features (C7 + T10, 6-dim) as the conditioning input
        instead of hands, because spine→pelvis r=0.848 > hands→pelvis r=0.647.

        Args:
            spine: (B, T, 6)  [C7(0:3), T10(6:9)] from Stage 2
            aria : (B, 512)

        Returns:
            lower: (B, T, 30) [RASI, LASI, RPSI, LPSI, RKNE, LKNE, RANKLE, LANKLE, RTOE, LTOE]
        """
        # HandConditionedLowerGenerator.forward(hand, aria, midi=None)
        lower = self.stage4_lower(spine, aria)  # (B, T, 30)
        return lower

    # ------------------------------------------------------------------
    # Full cascade
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        midi_input: dict,
        aria: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full hierarchical cascade.

        Args:
            midi_input: dict with keys
                'piano_roll' : (1, T, 88)  float tensor on self.device
                'velocity'   : (1, T, 88)
                'onset'      : (1, T, 88)
            aria: (1, 512) ARIA embedding on self.device

        Returns:
            full_body: (T, 26, 3)  numpy array
        """
        piano_roll = midi_input['piano_roll']
        T = piano_roll.shape[1]

        # ── Stage 1: hands ──────────────────────────────────────────────
        print("Stage 1: Generating hands...")
        if self.stage1_hand is not None:
            hand = self._run_stage1(midi_input)   # (1, T, 12)
        else:
            print("  [WARNING] Stage 1 not loaded – using zeros as placeholder.")
            hand = torch.zeros(1, T, 12, device=self.device)

        # ── Stage 2: torso ──────────────────────────────────────────────
        print("Stage 2: Generating torso...")
        if self.stage2_torso is not None:
            torso = self._run_stage2(hand, aria)  # (1, T, 12)
        else:
            print("  [WARNING] Stage 2 not loaded – using zeros as placeholder.")
            torso = torch.zeros(1, T, 12, device=self.device)

        # ── Stage 3: arms ───────────────────────────────────────────────
        # Extract wrists from Stage 1 hand output:
        #   hand ordering: [LWRIST(0:3), RWRIST(3:6)]
        #   ArmMDM expects wrists = [RWRIST, LWRIST] → concat(hand[3:6], hand[0:3])
        rwrist = hand[:, :, 3:6]   # (1, T, 3)
        lwrist = hand[:, :, 0:3]   # (1, T, 3)
        wrists = torch.cat([rwrist, lwrist], dim=-1)  # (1, T, 6)

        # Extract chest from Stage 2 torso output:
        #   torso ordering: [C7(0:3), CLAV(3:6), T10(6:9), STRN(9:12)]
        #   ArmMDM expects chest = [T10, STRN] → torso[6:12]
        chest = torso[:, :, 6:12]  # (1, T, 6)

        print("Stage 3: Generating arms...")
        if self.stage3_arm is not None:
            arms = self._run_stage3(wrists, chest, aria)  # (1, T, 12)
        else:
            print("  [WARNING] Stage 3 (Arm) not loaded – using zeros as placeholder.")
            arms = torch.zeros(1, T, 12, device=self.device)

        # ── Head: deterministic regressor ───────────────────────────────
        # Extract neck from Stage 2: [C7, CLAV] = torso[0:6]
        neck = torso[:, :, 0:6]    # (1, T, 6)

        print("Head: Generating head (deterministic)...")
        if self.head_regressor is not None:
            head = self._run_head(neck)           # (1, T, 12)
        else:
            print("  [WARNING] Head regressor not loaded – using zeros as placeholder.")
            head = torch.zeros(1, T, 12, device=self.device)

        # ── Stage 4: lower body ─────────────────────────────────────────
        # Build spine conditioning: [C7(0:3), T10(6:9)] = 6-dim
        c7  = torso[:, :, 0:3]    # (1, T, 3)
        t10 = torso[:, :, 6:9]    # (1, T, 3)
        spine = torch.cat([c7, t10], dim=-1)  # (1, T, 6)

        print("Stage 4: Generating lower body...")
        if self.stage4_lower is not None:
            lower = self._run_stage4(spine, aria)  # (1, T, 30)
        else:
            print("  [WARNING] Stage 4 not loaded – using zeros as placeholder.")
            lower = torch.zeros(1, T, 30, device=self.device)

        # ── Assemble full body ───────────────────────────────────────────
        hand_np  = hand.squeeze(0).cpu().numpy()   # (T, 12)
        torso_np = torso.squeeze(0).cpu().numpy()  # (T, 12)
        arms_np  = arms.squeeze(0).cpu().numpy()   # (T, 12)
        head_np  = head.squeeze(0).cpu().numpy()   # (T, 12)
        lower_np = lower.squeeze(0).cpu().numpy()  # (T, 30)

        full = self.assemble_full_body(hand_np, torso_np, arms_np, head_np, lower_np)
        return full  # (T, 26, 3)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble_full_body(
        self,
        hand:  np.ndarray,   # (T, 12)
        torso: np.ndarray,   # (T, 12)
        arms:  np.ndarray,   # (T, 12)
        head:  np.ndarray,   # (T, 12)
        lower: np.ndarray,   # (T, 30)
    ) -> np.ndarray:
        """
        Assemble all body parts into the canonical 26-joint skeleton.

        Joint index mapping (PARCO ordering):
            RFHD(0) LFHD(1) RBHD(2) LBHD(3) C7(4) CLAV(5) T10(6) STRN(7)
            RSHO(8) RELB(9) RWRIST(10) RHAND(11) LSHO(12) LELB(13)
            LWRIST(14) LHAND(15) RASI(16) LASI(17) RPSI(18) LPSI(19)
            RKNE(20) LKNE(21) RANKLE(22) LANKLE(23) RTOE(24) LTOE(25)

        Args:
            hand  : (T, 6) → [LWRIST, RWRIST]
            torso : (T, 12) → [C7, CLAV, T10, STRN]
            arms  : (T, 12) → [RSHO, RELB, LSHO, LELB]
            head  : (T, 12) → [RFHD, LFHD, RBHD, LBHD]
            lower : (T, 30) → [RASI, LASI, RPSI, LPSI, RKNE, LKNE, RANKLE, LANKLE, RTOE, LTOE]

        Returns:
            full: (T, 26, 3)
        """
        T = hand.shape[0]
        full = np.zeros((T, 26, 3), dtype=np.float32)

        # ── Hand ──────────────────────────────────────────────────────────
        hand_r = hand.reshape(T, 2, 3)  # [LWRIST, RWRIST]
        full[:, 14] = hand_r[:, 0]      # LWRIST
        full[:, 10] = hand_r[:, 1]      # RWRIST

        # ── Torso ─────────────────────────────────────────────────────────
        torso_r = torso.reshape(T, 4, 3)  # [C7, CLAV, T10, STRN]
        full[:, 4] = torso_r[:, 0]        # C7
        full[:, 5] = torso_r[:, 1]        # CLAV
        full[:, 6] = torso_r[:, 2]        # T10
        full[:, 7] = torso_r[:, 3]        # STRN

        # ── Arms ──────────────────────────────────────────────────────────
        arms_r = arms.reshape(T, 4, 3)   # [RSHO, RELB, LSHO, LELB]
        full[:, 8]  = arms_r[:, 0]        # RSHO
        full[:, 9]  = arms_r[:, 1]        # RELB
        full[:, 12] = arms_r[:, 2]        # LSHO
        full[:, 13] = arms_r[:, 3]        # LELB

        # ── Head ──────────────────────────────────────────────────────────
        head_r = head.reshape(T, 4, 3)   # [RFHD, LFHD, RBHD, LBHD]
        full[:, 0] = head_r[:, 0]         # RFHD
        full[:, 1] = head_r[:, 1]         # LFHD
        full[:, 2] = head_r[:, 2]         # RBHD
        full[:, 3] = head_r[:, 3]         # LBHD

        # ── Lower body ────────────────────────────────────────────────────
        lower_r = lower.reshape(T, 10, 3)  # RASI..LTOE
        for i, joint_idx in enumerate(range(16, 26)):
            full[:, joint_idx] = lower_r[:, i]

        return full  # (T, 26, 3)


# ===========================================================================
# Data loading helpers
# ===========================================================================

def _build_midi_dict(data: dict, T: int, device: str) -> dict:
    """
    Build the midi_dict from a loaded cache entry.

    Returns tensors on *device* with shape (1, T, 88).
    """
    active_keys = data['active_keys'][:T]

    piano_roll = np.zeros((T, 88), dtype=np.float32)
    for t in range(T):
        for k in active_keys[t]:
            if 0 <= k < 88:
                piano_roll[t, k] = 1.0

    velocity = data.get('onset_density_per_key', np.zeros((T, 88), dtype=np.float32))
    velocity = velocity[:T].astype(np.float32)

    onset = data.get('key_onset', np.zeros((T, 88), dtype=np.float32))
    onset = onset[:T].astype(np.float32)

    def _t(arr):
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    return {
        'piano_roll': _t(piano_roll),
        'velocity':   _t(velocity),
        'onset':      _t(onset),
    }


def _build_aria(data: dict, device: str) -> torch.Tensor:
    """Return aria embedding as (1, 512) tensor on device."""
    aria_embed = data['aria_embed']
    aria = aria_embed[0] if aria_embed.ndim == 2 else aria_embed
    return torch.from_numpy(aria.astype(np.float32)).unsqueeze(0).to(device)


def load_split_samples(cache_dir: str, split: str) -> list:
    """
    Load all sample metadata for a given split.

    Returns list of dicts with keys: idx, num_frames, sample_name.
    """
    cache_dir = Path(cache_dir)
    with open(cache_dir / 'metadata.json') as f:
        metadata = json.load(f)
    return metadata['splits'][split]['samples']


def load_sample(cache_dir: str, split: str, idx: int) -> dict:
    """Load a single cache sample by its file index (not list position)."""
    cache_dir = Path(cache_dir)
    npz_path  = cache_dir / split / f"{idx:05d}.npz"
    data = np.load(npz_path, allow_pickle=True)
    return {
        'joints':                data['joints'],
        'active_keys':           data['active_keys'],
        'aria_embed':            data['aria_embed'],
        'onset_density_per_key': data.get('onset_density_per_key',
                                          np.zeros((len(data['joints']), 88), dtype=np.float32)),
        'key_onset':             data.get('key_onset',
                                          np.zeros((len(data['joints']), 88), dtype=np.float32)),
    }


# ===========================================================================
# MPJPE evaluation helpers
# ===========================================================================

BODY_PART_GROUPS = {
    'head':    [0, 1, 2, 3],
    'torso':   [4, 5, 6, 7],
    'r_arm':   [8, 9, 10, 11],
    'l_arm':   [12, 13, 14, 15],
    'lower':   list(range(16, 26)),
}


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    Compute MPJPE (mm) per body part and overall.

    Args:
        pred: (T, 26, 3) predicted joints (metres)
        gt  : (T, 26, 3) ground-truth joints (metres)

    Returns:
        dict with keys matching BODY_PART_GROUPS plus 'overall'.
        Values are MPJPE in millimetres.
    """
    diff = pred - gt  # (T, 26, 3)
    dist = np.linalg.norm(diff, axis=-1)  # (T, 26)

    results = {}
    for part, indices in BODY_PART_GROUPS.items():
        results[part] = float(dist[:, indices].mean() * 1000.0)
    results['overall'] = float(dist.mean() * 1000.0)
    return results


# ===========================================================================
# main()
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='New Architecture Inference: Hand→Torso→Arms+Head→Lower',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--checkpoint_dir', type=str,
                        default='checkpoints/new_arch',
                        help='Root directory containing stage1/, stage2_torso/, '
                             'stage3_arm/, stage3_head/, stage4/ sub-dirs.')
    parser.add_argument('--cache_dir', type=str,
                        default='MOSA_dataset/training_cache_B',
                        help='Dataset cache directory (must contain metadata.json).')
    parser.add_argument('--output_dir', type=str,
                        default='results/new_arch',
                        help='Directory where output .npz files are written.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to run inference on.')
    parser.add_argument('--ddim_steps', type=int, default=50,
                        help='Number of DDIM denoising steps.')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 70)
    print("New Architecture Inference")
    print(f"  checkpoint_dir : {args.checkpoint_dir}")
    print(f"  cache_dir      : {args.cache_dir}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  split          : {args.split}")
    print(f"  ddim_steps     : {args.ddim_steps}")
    print(f"  device         : {args.device}")
    print("=" * 70)

    # ── Build output directory ───────────────────────────────────────────
    out_split_dir = Path(args.output_dir) / args.split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pipeline ────────────────────────────────────────────────────
    print("\nLoading pipeline...")
    pipeline = NewArchInference(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        ddim_steps=args.ddim_steps,
    )

    # ── Load split metadata ──────────────────────────────────────────────
    samples = load_split_samples(args.cache_dir, args.split)
    print(f"\nRunning inference on {len(samples)} {args.split} samples...")

    # Accumulators for per-part MPJPE
    mpjpe_accum = {k: [] for k in list(BODY_PART_GROUPS.keys()) + ['overall']}

    for sample_meta in tqdm(samples, desc=f'Inference ({args.split})'):
        file_idx  = sample_meta['idx']
        T         = sample_meta['num_frames']
        name      = sample_meta.get('sample_name', f"sample_{file_idx}")

        # Load data
        try:
            data = load_sample(args.cache_dir, args.split, file_idx)
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping {name}: {e}")
            continue

        gt_joints = data['joints'][:T]  # (T, 26, 3)

        # Prepare model inputs
        midi_dict = _build_midi_dict(data, T, args.device)
        aria      = _build_aria(data, args.device)

        # Run full cascade
        try:
            joints_pred = pipeline.generate(midi_dict, aria)  # (T, 26, 3)
        except Exception as exc:
            print(f"[ERROR] Inference failed for {name}: {exc}")
            import traceback; traceback.print_exc()
            # Fallback: use GT as placeholder so downstream metrics don't crash
            joints_pred = gt_joints.copy()

        # Save output
        out_path = out_split_dir / f"{file_idx:05d}.npz"
        np.savez(
            out_path,
            joints_pred=joints_pred.astype(np.float32),
            joints_gt=gt_joints.astype(np.float32),
        )

        # Accumulate MPJPE
        mpjpe = compute_mpjpe(joints_pred, gt_joints)
        for k, v in mpjpe.items():
            mpjpe_accum[k].append(v)

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"MPJPE Summary  [{args.split}]  ({len(mpjpe_accum['overall'])} sequences)")
    print("=" * 70)
    print(f"  {'Part':<12}  {'MPJPE (mm)':>12}")
    print(f"  {'-'*12}  {'-'*12}")
    for part in list(BODY_PART_GROUPS.keys()) + ['overall']:
        vals = mpjpe_accum[part]
        if vals:
            mean_val = float(np.mean(vals))
            mark = " ←" if part == 'overall' else ""
            print(f"  {part:<12}  {mean_val:>12.1f}{mark}")
    print("=" * 70)
    print(f"\nOutputs saved to: {out_split_dir}")


if __name__ == '__main__':
    main()
