"""
Constants for MOSA dataset and motion representation

Simplified skeleton: 26 joints
- Original 30 body markers → 26 simplified joints
- Preserves individual markers for head (4), neck (2), chest (2), pelvis (4)
- Only wrist uses averaging (RWRA+RWRB, LWRA+LWRB) - anatomically recoverable
"""

# ============================================================
# RAW MOSA MARKERS (for data loading)
# ============================================================
RAW_MARKER_NAMES = [
    'RFHD', 'LFHD', 'RBHD', 'LBHD',  # Head markers (0-3)
    'C7', 'T10', 'CLAV', 'STRN',      # Torso markers (4-7)
    'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN',  # Right arm (8-12)
    'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN',  # Left arm (13-17)
    'RASI', 'LASI', 'RPSI', 'LPSI',  # Pelvis (18-21)
    'RKNE', 'RHEE', 'RTOE', 'RANK',  # Right leg (22-25)
    'LKNE', 'LHEE', 'LTOE', 'LANK',  # Left leg (26-29)
]

# Legacy aliases for compatibility
MARKER_NAMES = RAW_MARKER_NAMES
UPPER_BODY_MARKERS = RAW_MARKER_NAMES[:18]  # Head + Torso + Arms
RIGHT_WRIST_MARKERS = ['RWRA', 'RWRB']
LEFT_WRIST_MARKERS = ['LWRA', 'LWRB']

# ============================================================
# SIMPLIFIED SKELETON (26 joints)
# Preserves all non-recoverable information
# ============================================================
FULL_BODY_JOINTS = [
    # Head (4) - individual markers
    'RFHD',           # 0: Right front head
    'LFHD',           # 1: Left front head
    'RBHD',           # 2: Right back head
    'LBHD',           # 3: Left back head
    # Neck (2) - split for front/back info
    'C7',             # 4: Back neck (7th cervical vertebra)
    'CLAV',           # 5: Front clavicle
    # Chest (2) - split for front/back info
    'T10',            # 6: Back thoracic (10th thoracic vertebra)
    'STRN',           # 7: Front sternum
    # Right arm (4)
    'RSHO',           # 8: Right shoulder
    'RELB',           # 9: Right elbow
    'RWRIST',         # 10: Right wrist (avg of RWRA, RWRB - recoverable)
    'RHAND',          # 11: Right hand (RFIN)
    # Left arm (4)
    'LSHO',           # 12: Left shoulder
    'LELB',           # 13: Left elbow
    'LWRIST',         # 14: Left wrist (avg of LWRA, LWRB - recoverable)
    'LHAND',          # 15: Left hand (LFIN)
    # Pelvis (4) - individual markers for rotation info
    'RASI',           # 16: Right anterior superior iliac spine
    'LASI',           # 17: Left anterior superior iliac spine
    'RPSI',           # 18: Right posterior superior iliac spine
    'LPSI',           # 19: Left posterior superior iliac spine
    # Right leg (3)
    'RKNE',           # 20: Right knee
    'RANKLE',         # 21: Right ankle
    'RTOE',           # 22: Right toe
    # Left leg (3)
    'LKNE',           # 23: Left knee
    'LANKLE',         # 24: Left ankle
    'LTOE',           # 25: Left toe
]

NUM_FULL_BODY_JOINTS = len(FULL_BODY_JOINTS)  # 26

# Joint name aliases for compatibility
JOINT_NAMES = FULL_BODY_JOINTS
NUM_JOINTS = NUM_FULL_BODY_JOINTS

# ============================================================
# JOINT INDICES
# ============================================================
# Head
RFHD_IDX = 0
LFHD_IDX = 1
RBHD_IDX = 2
LBHD_IDX = 3
# Neck
C7_IDX = 4
CLAV_IDX = 5
# Chest
T10_IDX = 6
STRN_IDX = 7
# Right arm
RSHO_IDX = 8
RELB_IDX = 9
RWRIST_IDX = 10
RHAND_IDX = 11
# Left arm
LSHO_IDX = 12
LELB_IDX = 13
LWRIST_IDX = 14
LHAND_IDX = 15
# Pelvis
RASI_IDX = 16
LASI_IDX = 17
RPSI_IDX = 18
LPSI_IDX = 19
# Right leg
RKNE_IDX = 20
RANKLE_IDX = 21
RTOE_IDX = 22
# Left leg
LKNE_IDX = 23
LANKLE_IDX = 24
LTOE_IDX = 25

# Legacy aliases for compatibility
HEAD_IDX = RFHD_IDX
NOSE_IDX = HEAD_IDX
NECK_IDX = C7_IDX  # Use C7 as neck reference
CHEST_IDX = T10_IDX  # Use T10 as chest reference
FB_RWRIST_IDX = RWRIST_IDX
FB_LWRIST_IDX = LWRIST_IDX
RFIN_IDX = RHAND_IDX
LFIN_IDX = LHAND_IDX
# Hip aliases (computed from pelvis markers)
RHIP_IDX = RASI_IDX  # Approximate
LHIP_IDX = LASI_IDX  # Approximate
PELVIS_IDX = RASI_IDX  # Approximate center

# ============================================================
# SKELETON EDGES (30 edges)
# ============================================================
FULL_BODY_EDGES = [
    # === HEAD box (4 markers) ===
    (0, 1),   # RFHD -> LFHD (front face)
    (2, 3),   # RBHD -> LBHD (back head)
    (0, 2),   # RFHD -> RBHD (right side)
    (1, 3),   # LFHD -> LBHD (left side)
    
    # === HEAD -> NECK (back of head to C7) ===
    (2, 4),   # RBHD -> C7
    (3, 4),   # LBHD -> C7
    
    # === SPINE (back chain) ===
    (4, 6),   # C7 -> T10 (upper to mid spine)
    
    # === FRONT TORSO (sternum chain) ===
    (5, 7),   # CLAV -> STRN
    
    # === SHOULDERS (from both C7 and CLAV) ===
    (4, 8),   # C7 -> RSHO (back neck to shoulder)
    (4, 12),  # C7 -> LSHO
    (5, 8),   # CLAV -> RSHO (front clavicle to shoulder)
    (5, 12),  # CLAV -> LSHO
    
    # === RIGHT ARM chain ===
    (8, 9),   # RSHO -> RELB
    (9, 10),  # RELB -> RWRIST
    (10, 11), # RWRIST -> RHAND
    
    # === LEFT ARM chain ===
    (12, 13), # LSHO -> LELB
    (13, 14), # LELB -> LWRIST
    (14, 15), # LWRIST -> LHAND
    
    # === TORSO -> PELVIS ===
    (6, 18),  # T10 -> RPSI (back spine to back pelvis)
    (6, 19),  # T10 -> LPSI
    (7, 16),  # STRN -> RASI (front torso to front pelvis)
    (7, 17),  # STRN -> LASI
    
    # === PELVIS box (4 markers) ===
    (16, 17), # RASI -> LASI (front pelvis line)
    (18, 19), # RPSI -> LPSI (back pelvis line)
    (16, 18), # RASI -> RPSI (right side)
    (17, 19), # LASI -> LPSI (left side)
    
    # === RIGHT LEG (from pelvis to foot) ===
    (16, 20), # RASI -> RKNE
    (18, 20), # RPSI -> RKNE
    (20, 21), # RKNE -> RANKLE
    (21, 22), # RANKLE -> RTOE
    
    # === LEFT LEG (from pelvis to foot) ===
    (17, 23), # LASI -> LKNE
    (19, 23), # LPSI -> LKNE
    (23, 24), # LKNE -> LANKLE
    (24, 25), # LANKLE -> LTOE
]

# ============================================================
# BODY PART GROUPING
# ============================================================
BODY_PARTS = {
    'head': {
        'joints': ['RFHD', 'LFHD', 'RBHD', 'LBHD'],
        'indices': [0, 1, 2, 3],
    },
    'neck': {
        'joints': ['C7', 'CLAV'],
        'indices': [4, 5],
    },
    'chest': {
        'joints': ['T10', 'STRN'],
        'indices': [6, 7],
    },
    'right_arm': {
        'joints': ['RSHO', 'RELB', 'RWRIST', 'RHAND'],
        'indices': [8, 9, 10, 11],
    },
    'left_arm': {
        'joints': ['LSHO', 'LELB', 'LWRIST', 'LHAND'],
        'indices': [12, 13, 14, 15],
    },
    'pelvis': {
        'joints': ['RASI', 'LASI', 'RPSI', 'LPSI'],
        'indices': [16, 17, 18, 19],
    },
    'right_leg': {
        'joints': ['RKNE', 'RANKLE', 'RTOE'],
        'indices': [20, 21, 22],
    },
    'left_leg': {
        'joints': ['LKNE', 'LANKLE', 'LTOE'],
        'indices': [23, 24, 25],
    },
}

# Part indices
HEAD_INDICES = BODY_PARTS['head']['indices']  # 4 joints
NECK_INDICES = BODY_PARTS['neck']['indices']  # 2 joints
CHEST_INDICES = BODY_PARTS['chest']['indices']  # 2 joints
TORSO_INDICES = NECK_INDICES + CHEST_INDICES  # 4 joints (neck + chest)
R_ARM_INDICES = BODY_PARTS['right_arm']['indices']  # 4 joints
L_ARM_INDICES = BODY_PARTS['left_arm']['indices']  # 4 joints
ARMS_INDICES = R_ARM_INDICES + L_ARM_INDICES  # 8 joints
PELVIS_INDICES = BODY_PARTS['pelvis']['indices']  # 4 joints
R_LEG_INDICES = BODY_PARTS['right_leg']['indices']  # 3 joints
L_LEG_INDICES = BODY_PARTS['left_leg']['indices']  # 3 joints
LEGS_INDICES = R_LEG_INDICES + L_LEG_INDICES  # 6 joints

# Upper body (head + neck + chest + arms = 16 joints)
UPPER_BODY_INDICES = HEAD_INDICES + TORSO_INDICES + ARMS_INDICES

# Lower body (pelvis + legs = 10 joints)
LOWER_BODY_INDICES = PELVIS_INDICES + LEGS_INDICES

# ============================================================
# DATA-DRIVEN DECOMPOSITION INDICES
# ============================================================
# Based on motion-music correlation analysis (body_decomposition_analysis.json):
#   MIDI→hands r=0.55-0.61 (direct), MIDI→spine r=0.265 (hand-mediated)
#   head-C7 r=0.934 (rigid follow), spine→pelvis r=0.848 (cascade)
#   elbow-wrist r=0.946, elbow-shoulder r=0.770 (geometry constrained)

# Stage 2: Torso (neck + chest), driven by hands
# Note: TORSO_INDICES already defined above as NECK_INDICES + CHEST_INDICES = [4, 5, 6, 7]
# Reaffirm explicit ordering: [C7(4), CLAV(5), T10(6), STRN(7)]
NUM_TORSO_JOINTS = 4

# Stage 3: Arm (shoulders + elbows), conditioned on wrists + chest
ARM_INDICES = [RSHO_IDX, RELB_IDX, LSHO_IDX, LELB_IDX]  # [8, 9, 12, 13]
NUM_ARM_JOINTS = 4

# Head: rigid follower of C7/CLAV (r=0.934), non-diffusion regressor
HEAD_FOLLOWER_INDICES = HEAD_INDICES  # [0, 1, 2, 3]
NUM_HEAD_JOINTS = 4

# Chest subset (used as ArmMDM conditioning anchor)
CHEST_INDICES_FOR_ARM = [T10_IDX, STRN_IDX]  # [6, 7]

# Spine subset (used as LowerFiLM conditioning: r_spine_pelvis=0.848)
SPINE_INDICES_FOR_LOWER = [C7_IDX, T10_IDX]  # [4, 6]

# Full decomposition (26 joints total):
# S1=4 + S2=4 + S3=4 + Head=4 + S4=10 = 26
DECOMP_STAGE1_INDICES = [LWRIST_IDX, RWRIST_IDX]  # wrists
DECOMP_STAGE2_INDICES = TORSO_INDICES                                      # torso
DECOMP_STAGE3_INDICES = ARM_INDICES                                        # arms
DECOMP_HEAD_INDICES   = HEAD_INDICES                                       # head
DECOMP_STAGE4_INDICES = LOWER_BODY_INDICES                                 # lower

# ============================================================
# ParCo 6-PART DIVISION (for s30_train_parco.py)
# Following ParCo (ECCV 2024) part decomposition
# ============================================================
PARCO_PARTS = {
    'Backbone': {
        'joints': ['RFHD', 'LFHD', 'RBHD', 'LBHD', 'C7', 'CLAV', 'T10', 'STRN'],
        'indices': [0, 1, 2, 3, 4, 5, 6, 7],
        'num_joints': 8,
        'input_dim': 6,  # both shoulders
    },
    'R_Arm': {
        'joints': ['RSHO', 'RELB', 'RWRIST', 'RHAND'],
        'indices': [8, 9, 10, 11],
        'num_joints': 4,
        'input_dim': 3,  # right wrist
    },
    'L_Arm': {
        'joints': ['LSHO', 'LELB', 'LWRIST', 'LHAND'],
        'indices': [12, 13, 14, 15],
        'num_joints': 4,
        'input_dim': 3,  # left wrist
    },
    'Root': {
        'joints': ['RASI', 'LASI', 'RPSI', 'LPSI'],
        'indices': [16, 17, 18, 19],
        'num_joints': 4,
        'input_dim': 6,  # T10 + STRN
    },
    'R_Leg': {
        'joints': ['RKNE', 'RANKLE', 'RTOE'],
        'indices': [20, 21, 22],
        'num_joints': 3,
        'input_dim': 6,  # RASI + RPSI
    },
    'L_Leg': {
        'joints': ['LKNE', 'LANKLE', 'LTOE'],
        'indices': [23, 24, 25],
        'num_joints': 3,
        'input_dim': 6,  # LASI + LPSI
    },
}

PARCO_PART_NAMES = ['Backbone', 'R_Arm', 'L_Arm', 'Root', 'R_Leg', 'L_Leg']

# ParCo part indices (for slicing)
PARCO_BACKBONE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 joints
PARCO_R_ARM_INDICES = [8, 9, 10, 11]  # 4 joints
PARCO_L_ARM_INDICES = [12, 13, 14, 15]  # 4 joints
PARCO_ROOT_INDICES = [16, 17, 18, 19]  # 4 joints
PARCO_R_LEG_INDICES = [20, 21, 22]  # 3 joints
PARCO_L_LEG_INDICES = [23, 24, 25]  # 3 joints

# ParCo joint counts
PARCO_NUM_BACKBONE = 8
PARCO_NUM_R_ARM = 4
PARCO_NUM_L_ARM = 4
PARCO_NUM_ROOT = 4
PARCO_NUM_R_LEG = 3
PARCO_NUM_L_LEG = 3

# ============================================================
# TRAINING PARTS (for s30_train_body_parts.py) - 3 Parts (legacy)
# ============================================================
# Part 1: Head + Torso (8 joints: indices 0-7)
TRAIN_HEAD_TORSO_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]

# Part 2: Arms (8 joints: indices 8-15)  
TRAIN_ARMS_INDICES = [8, 9, 10, 11, 12, 13, 14, 15]

# Part 3: Lower body (10 joints: indices 16-25)
TRAIN_LOWER_INDICES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

# Training part counts
NUM_HEAD_TORSO_JOINTS = 8
NUM_ARMS_JOINTS = 8  
NUM_LOWER_JOINTS = 10

# Arms-only edges (for arms generator, local indices)
ARMS_EDGES = [
    (0, 1),   # RSHO -> RELB
    (1, 2),   # RELB -> RWRIST
    (2, 3),   # RWRIST -> RHAND
    (4, 5),   # LSHO -> LELB
    (5, 6),   # LELB -> LWRIST
    (6, 7),   # LWRIST -> LHAND
]

# ============================================================
# MAPPING FROM RAW MARKERS TO SIMPLIFIED JOINTS
# Only wrist uses averaging (anatomically recoverable)
# ============================================================
JOINT_MAPPING = {
    # Head (4 individual markers)
    'RFHD': {'type': 'single', 'marker': 'RFHD'},
    'LFHD': {'type': 'single', 'marker': 'LFHD'},
    'RBHD': {'type': 'single', 'marker': 'RBHD'},
    'LBHD': {'type': 'single', 'marker': 'LBHD'},
    # Neck (2 individual markers - preserves front/back)
    'C7': {'type': 'single', 'marker': 'C7'},
    'CLAV': {'type': 'single', 'marker': 'CLAV'},
    # Chest (2 individual markers - preserves front/back)
    'T10': {'type': 'single', 'marker': 'T10'},
    'STRN': {'type': 'single', 'marker': 'STRN'},
    # Arms (wrist averaged - recoverable)
    'RSHO': {'type': 'single', 'marker': 'RSHO'},
    'RELB': {'type': 'single', 'marker': 'RELB'},
    'RWRIST': {'type': 'average', 'markers': ['RWRA', 'RWRB']},
    'RHAND': {'type': 'single', 'marker': 'RFIN'},
    'LSHO': {'type': 'single', 'marker': 'LSHO'},
    'LELB': {'type': 'single', 'marker': 'LELB'},
    'LWRIST': {'type': 'average', 'markers': ['LWRA', 'LWRB']},
    'LHAND': {'type': 'single', 'marker': 'LFIN'},
    # Pelvis (4 individual markers - preserves rotation)
    'RASI': {'type': 'single', 'marker': 'RASI'},
    'LASI': {'type': 'single', 'marker': 'LASI'},
    'RPSI': {'type': 'single', 'marker': 'RPSI'},
    'LPSI': {'type': 'single', 'marker': 'LPSI'},
    # Legs
    'RKNE': {'type': 'single', 'marker': 'RKNE'},
    'RANKLE': {'type': 'single', 'marker': 'RANK'},
    'RTOE': {'type': 'single', 'marker': 'RTOE'},
    'LKNE': {'type': 'single', 'marker': 'LKNE'},
    'LANKLE': {'type': 'single', 'marker': 'LANK'},
    'LTOE': {'type': 'single', 'marker': 'LTOE'},
}

# ============================================================
# PIANO & TRAINING SETTINGS
# ============================================================

# Piano key ranges (MIDI numbers)
PIANO_KEY_MIN = 21   # A0
PIANO_KEY_MAX = 108  # C8
NUM_PIANO_KEYS = PIANO_KEY_MAX - PIANO_KEY_MIN + 1  # 88 keys

# Key bins for wrist prior (based on tipiano approach - 8 bins)
KEY_BINS = [
    (0, 15, 'very_low_1'),    # A0-B1: very low pitch 1
    (15, 25, 'very_low_2'),   # C2-A2: very low pitch 2
    (25, 35, 'low'),          # A#2-G3: low pitch
    (35, 45, 'mid_low'),      # G#3-E4: mid-low pitch
    (45, 55, 'center'),       # F4-C#5: center (around middle C)
    (55, 65, 'mid_high'),     # D5-A5: mid-high pitch
    (65, 75, 'high'),         # A#5-F6: high pitch
    (75, 88, 'very_high'),    # F#6-C8: very high pitch
]

# Frame rate (undersampled from original 120fps)
FPS = 60  # Training frame rate (downsampled from 120fps)

# Data unit: all motion data is stored in METERS
# Raw MOSA data is in mm, converted to m during preprocessing (s05_preprocess_motion.py)
DATA_UNIT = 'meters'
MM_TO_M = 0.001
M_TO_MM = 1000.0

# Window settings for training
WINDOW_SIZE_SEC = 8.0
STRIDE_SIZE_SEC = 4.0

# Henle difficulty categories
DIFFICULTY_CATEGORIES = {
    'easy': [3, 4],
    'medium': [5, 6],
    'hard': [7, 8, 9]
}
