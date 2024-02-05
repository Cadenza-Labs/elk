elk sweep --num_gpus 4 --models EleutherAI/pythia-12b EleutherAI/pythia-6.9b EleutherAI/pythia-12b --datasets imdb amazon_polarity meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf openai-community/gpt2-medium openai-community/gpt2-large "mistralai/Mistral-7B-v0.1"   --datasets imdb amazon_polarity   --skip_transfer_eval --net ccs --norm cluster  --cluster_algo kmeans  --k_clusters 100
git rev-parse cadenza/k-means

# "mistralai/Mixtral-8x7B-v0.1"
