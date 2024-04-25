# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/explicit_opinion/train/ --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval  --num_gpus 6 # --disable_cache
git rev-parse cadenza/k-means

# # cluster norm
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2 --net crc --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 8 # --disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/8 --net crc --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 8 #--disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/32 --net crc --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 8 #--disable_cache

# # deepmind no templates norm
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/2 --net crc --norm burns --cluster_algo None --models ""mistralai/Mistral-7B-v0.1"" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/8 --net crc --norm burns --cluster_algo None --models ""mistralai/Mistral-7B-v0.1"" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/32 --net crc --norm burns --cluster_algo None --models ""mistralai/Mistral-7B-v0.1"" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/64 --net crc --norm burns --cluster_algo None --models ""mistralai/Mistral-7B-v0.1"" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache
# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/random/train/imdb/128 --net crc --norm burns --cluster_algo None --models ""mistralai/Mistral-7B-v0.1"" --datasets imdb --skip_transfer_eval  --num_gpus 8 # --disable_cache


CUDA_VISIBLE_DEVICES=0,1,6,7 elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/random/train/boolq/64 --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets super_glue:boolq --skip_transfer_eval   # --disable_cache
