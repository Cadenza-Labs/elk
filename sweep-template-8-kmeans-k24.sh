#export OMP_NUM_THREADS=20
#echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
elk sweep --template_path /home/laurito/elk/scripts/templates/imdb/8/ --net ccs --norm cluster --cluster_algo kmeans --k_clusters 24 --models "mistralai/Mistral-7B-v0.1" --datasets imdb --skip_transfer_eval --num_gpus 6 # --disable_cache
git rev-parse cadenza/k-means
