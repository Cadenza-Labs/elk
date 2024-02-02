elk sweep --num_gpus 4 --models meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf openai-community/gpt2-medium openai-community/gpt2-large "mistralai/Mistral-7B-v0.1" "mistralai/Mixtral-8x7B-v0.1"   --datasets imdb amazon_polarity   --skip_transfer_eval --net ccs --norm cluster  --cluster_algo kmeans  --k_clusters 100
git rev-parse cadenza/k-means
