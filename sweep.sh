elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/implicit_opinion/biased --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets "imdb" --skip_transfer_eval  --num_gpus 2 --disable_cache
# elk sweep --net.seed 42 --template_path /home/laurito/elk/scripts/templates/implicit_opinion/biased --net ccs --norm cluster --cluster_algo HDBSCAN --min_cluster_size 5 --models "mistralai/Mistral-7B-v0.1" --datasets "imdb" --skip_transfer_eval  --num_gpus 7 # --disable_cache
git rev-parse cadenza/k-means
