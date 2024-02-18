elk sweep --template_path /home/laurito/elk/scripts/templates/imdb/128/ --net ccs --norm cluster --cluster_algo kmeans --k_clusters 256 --models EleutherAI/pythia-12b EleutherAI/pythia-6.9b EleutherAI/pythia-12b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf openai-community/gpt2-medium openai-community/gpt2-large "mistralai/Mistral-7B-v0.1" --datasets imdb   --skip_transfer_eval  --num_gpus 2 --disable_cache
git rev-parse cadenza/k-means