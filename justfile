[confirm]
setup: sync download_ckpts

sync:
    uv sync
    uv sync --extra compile

download_sam_ckpt:
    #!/usr/bin/env bash
    mkdir ./ckpts
    echo "Downloading SAM checkpoint..."
    curl -L "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" -o ./ckpts/sam_vit_h_4b8939.pth

download_masqclip_ckpt:
    #!/usr/bin/env bash
    mkdir -p ./ckpts/MasQCLIP
    echo "Downloading MasQCLIP checkpoint..."
    uvx gdown 1_Sjlx37R4iDTBh78A15shiFZobE0oN_B -O ./ckpts/MasQCLIP/base_novel.pth

download_ckpts: download_sam_ckpt download_masqclip_ckpt

source_path := ""

sfm:
    #!/usr/bin/env bash
    if [ -z "{{source_path}}" ]; then
        echo "Error: source_path is not set."
        exit 1
    fi
    uv run python opensplat3d/data/preprocessing/sfm.py "{{source_path}}" --txt


mask_extraction:
    #!/usr/bin/env bash
    if [ -z "{{source_path}}" ]; then
        echo "Error: source_path is not set."
        exit 1
    fi
    uv run python opensplat3d/masks/extract_sam_masks.py "{{source_path}}" --sort score

[confirm]
preprocess: sfm mask_extraction
