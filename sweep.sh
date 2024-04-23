# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/explicit_opinion/train/ --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval  --num_gpus 6 # --disable_cache
# git rev-parse cadenza/k-means

# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/personas/train/imdb --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 2 --disable_cache
# elk sweep --template_path /home/laurito/elk/scripts/templates/random/train/imdb/1 --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 2  --disable_cache

# elk elicit "mistralai/Mistral-7B-v0.1" imdb --template_path /home/laurito/elk/scripts/templates/random/train/imdb/1 --norm burns --cluster_algo None --net ccs --num_gpus 7 #--disable_cache
elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2 --net crc --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2 --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
