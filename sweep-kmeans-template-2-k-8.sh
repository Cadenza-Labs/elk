elk sweep --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2/ --net ccs --norm cluster --cluster_algo kmeans --k_clusters 8 --models "mistralai/Mistral-7B-v0.1" --datasets imdb   --skip_transfer_eval  --num_gpus 6 --disable_cache
git rev-parse cadenza/k-means
