elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/train/imdb/2/ --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb   --skip_transfer_eval  --num_gpus 6 #--disable_cache
git rev-parse cadenza/k-means
