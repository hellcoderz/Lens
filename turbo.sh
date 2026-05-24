python inference.py \
    --repo_id "microsoft/Lens-Turbo" \
    --prompt "A cinematic mountain lake at sunrise, soft mist, detailed reflections" \
    --base_resolution 1440 --aspect_ratio 1:1 \
    --steps 4 --cfg 1.0 --n 1 --seed 42 \
    --out ./outputs