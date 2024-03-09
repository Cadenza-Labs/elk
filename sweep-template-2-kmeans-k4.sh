elk elicit EleutherAI/pythia-12b imdb \
    --template_path /home/laurito/elk/scripts/templates/train/imdb/2/ \
    --net ccs \
    --norm cluster \
    --cluster_algo kmeans \
    --k_clusters 4 \
    --num_gpus 6 \
    --disable_cache

git rev-parse cadenza/k-means
