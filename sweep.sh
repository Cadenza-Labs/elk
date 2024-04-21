elk sweep --net.seed 42 \
    --template_path /home/laurito/elk/scripts/templates/personas/train/claims/ \
    --datasets lauritowal/common_claims \
    --norm burns \
    --cluster_algo None \
    --net ccs \
    --models "mistralai/Mistral-7B-v0.1" \
    --skip_transfer_eval \
    --num_gpus 1 \
    --disable_cache
