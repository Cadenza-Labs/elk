elk sweep --net ccs --norm cluster --cluster_algo spectral --k_clusters 100 --models EleutherAI/pythia-12b EleutherAI/pythia-6.9b EleutherAI/pythia-12b meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf openai-community/gpt2-medium openai-community/gpt2-large "mistralai/Mistral-7B-v0.1" --datasets imdb amazon_polarity --skip_transfer_eval --num_gpus 4
git rev-parse cadenza/k-means
