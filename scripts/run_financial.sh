#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

models=("PatchTST" "Transformer" "TSMixer" "DLinear" "FiLM" "Koopa" "TiDE" "Autoformer" "Crossformer" "FEDformer" "Informer" "iTransformer" "Nonstationary_Transformer" "Reformer")

injection_modes=("unimodal" "last-additive" "first-additive" "middle-additive" "first-concat" "middle-concat" "last-concat" "film" "cfa" "gating" "orthogonal")
lr_multipliers=(0.05 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0)

llm_models=("BERT" "LLAMA3" "GPT2" "Doc2Vec")

datasets=(
  "./data/Financial|crude_oil_WTI.csv|daily"
  "./data/Financial|crude_oil_Brent.csv|daily"
)

seeds=(2021)

for llm_model in "${llm_models[@]}"; do
  llm_lower=${llm_model,,}  # lowercase
  for seed in "${seeds[@]}"; do
    for lr_multiplier in "${lr_multipliers[@]}"; do
      for injection_mode in "${injection_modes[@]}"; do
        for model_name in "${models[@]}"; do
          for dataset_info in "${datasets[@]}"; do
            IFS='|' read -r root_path data_path frequency <<< "$dataset_info"
            dataset_name=$(basename ${data_path} .csv)
            model_id=$(basename ${root_path})_${dataset_name}

            seq_lens=(30 60 90 120)
            pred_lens=(5 10 20 30)

            for seq_len in "${seq_lens[@]}"; do
              label_len=$((seq_len / 2))
              for pred_len in "${pred_lens[@]}"; do
                save_name="result_${model_name}_${dataset_name}_${llm_lower}_${injection_mode}"

                echo "Running $model_name on $dataset_name seq=$seq_len pred=$pred_len mode=$injection_mode lr=$lr_multiplier llm=$llm_lower"

                python -u run_financial.py \
                  --task_name long_term_forecast \
                  --is_training 1 \
                  --root_path $root_path \
                  --data_path $data_path \
                  --model_id ${model_id}_${seed}_${seq_len}_${pred_len}_fullLLM_0_${injection_mode}_${llm_lower} \
                  --model $model_name \
                  --data custom_financial \
                  --features M \
                  --seq_len $seq_len \
                  --label_len $label_len \
                  --pred_len $pred_len \
                  --des 'Exp' \
                  --seed $seed \
                  --type_tag "#F#" \
                  --text_len 4 \
                  --prompt_weight 0.1 \
                  --pool_type "avg" \
                  --save_name "$save_name" \
                  --llm_model $llm_model \
                  --huggingface_token 'NA' \
                  --use_text_integrated 1 \
                  --text_injection_mode $injection_mode \
                  --lr_multiplier $lr_multiplier
              done
            done
          done
        done
      done
    done
  done
done
