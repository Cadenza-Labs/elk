elk elicit meta-llama/Llama-2-13b-chat-hf imdb \
    --template_path /home/laurito/elk/scripts/templates/train/imdb/2/ \
    --net ccs \
    --norm cluster \
    --cluster_algo kmeans \
    --k_clusters 8 \
    --num_gpus 1 \
    # --disable_cache

git rev-parse cadenza/PCA_hovertext
