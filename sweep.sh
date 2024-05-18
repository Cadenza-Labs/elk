# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/personas/train/claims/ --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets lauritowal/claims --skip_transfer_eval --num_gpus 1 # --disable_cache
elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/personas/train/truthfulqa/ --net crc --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets lauritowal/truthful_qa --skip_transfer_eval --num_gpus 2 --disable_cache


# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/explicit_opinion/train/ --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval  --num_gpus 6 # --disable_cache
# git rev-parse cadenza/k-means

# elk sweep --template_path /home/laurito/elk/scripts/templates/random/train/imdb/1 --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 2  --disable_cache
# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/personas/train/imdb/2 --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 2 --disable_cache
elk sweep --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2 --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 8  # --disable_cache
