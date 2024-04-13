# elk sweep --net.seed 44 --template_path /home/laurito/elk/scripts/templates/implicit_opinion/train --net ccs --norm burns --cluster_algo None --models "mistralai/Mistral-7B-v0.1" --datasets fancyzhx/dbpedia_14 --skip_transfer_eval --num_gpus 2 --max_examples 3000 3000 --disable_cache
# git rev-parse cadenza/main


elk eval  /home/wombat_share/laurito/elk_reporters/sweeps/zealous-wilson/mistralai/Mistral-7B-v0.1/fancyzhx/dbpedia_14/ mistralai/Mistral-7B-v0.1 fancyzhx/dbpedia_14 --template_path /home/laurito/elk/scripts/templates/implicit_opinion/eval --num_gpus 1 --norm False
