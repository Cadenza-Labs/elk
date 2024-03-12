elk elicit mistralai/Mistral-7B-v0.1 imdb \
    --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2/ \
    --net ccs \
    --norm cluster \
    --cluster_algo kmeans \
    --k_clusters 6 \
    --num_gpus 6 \
    --disable_cache

git rev-parse cadenza/PCA_hovertext
