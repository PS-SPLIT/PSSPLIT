# -----------------
# DATASET ROOTS
# -----------------
# Replace these placeholders with the absolute paths to each dataset on your machine.
cifar_10_root        = '<PATH_TO_CIFAR10>'
cifar_100_root       = '<PATH_TO_CIFAR100>'
cub_root             = '<PATH_TO_CUB>'
car_root             = '<PATH_TO_STANFORD_CARS>/cars_{}/'
pets_root            = '<PATH_TO_PETS>'
aircraft_root        = '<PATH_TO_FGVC_AIRCRAFT>/fgvc-aircraft-2013b'
herbarium_dataroot   = '<PATH_TO_HERBARIUM_19>'
imagenet_root        = '<PATH_TO_IMAGENET1K>'

# OSR / SSB split directory (contains *_hier_5050.pkl etc).
osr_split_dir        = '<PATH_TO_SSB_SPLITS>'


# -----------------
# PRETRAINED BACKBONES
# -----------------
dino_pretrain_path2  = '<PATH_TO_DINOV2_VITB14_REG4_CHECKPOINT>'   # DINOv2 ViT-B/14 reg4 (default)
dino_pretrain_path   = '<PATH_TO_DINO_VITB16_PRETRAIN_CHECKPOINT>' # DINOv1 ViT-B/16 (legacy)
dino_v1_vitb16_path  = '<PATH_TO_DINO_VITB16_PRETRAIN_CHECKPOINT>'
warmup_pretrain_path = '<PATH_TO_GCD_WARMUP_CHECKPOINT>'           # unused in default pipeline


# -----------------
# OUTPUT DIRECTORIES (relative paths are fine)
# -----------------
feature_extract_dir = 'osr_novel_categories/extracted_features'   # extracted features land here
exp_root            = 'osr_novel_categories/'                     # logs and checkpoints land here


# -----------------
# iNat SUBSPECIES (optional)
# -----------------
inat_splits_csv      = '<PATH_TO_INAT_SUBSPECIES_SPLITS_CSV>'
inat_metadata_dir    = '<PATH_TO_INAT_SUBSPECIES_METADATA_DIR>'
inat_images_dir      = '<PATH_TO_INAT_SUBSPECIES_IMAGES_DIR>'


# -----------------
# AVES SMALL
# -----------------
aves_small_splits_csv   = '<PATH_TO_AVES_SMALL_SPLITS_CSV>'
aves_small_metadata_dir = '<PATH_TO_AVES_SMALL_METADATA_DIR>'
aves_small_images_dir   = '<PATH_TO_AVES_SMALL_IMAGES_DIR>'


# -----------------
# AVES FULL
# -----------------
aves_full_splits_csv    = '<PATH_TO_AVES_FULL_SPLITS_CSV>'
aves_full_metadata_dir  = '<PATH_TO_AVES_FULL_METADATA_DIR>'
aves_full_images_dir    = '<PATH_TO_AVES_FULL_IMAGES_DIR>'
