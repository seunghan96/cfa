#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


models=("PatchTST" "Transformer" "TSMixer" "DLinear" "FiLM" "Koopa" "TiDE" "Autoformer" "Crossformer" "FEDformer" "Informer" "iTransformer" "Nonstationary_Transformer" "Reformer")

injection_modes=("unimodal" "last-additive" "first-additive" "middle-additive" "first-concat" "middle-concat" "last-concat" "film" "cfa" "gating" "orthogonal")
lr_multipliers=(0.05 0.1 0.5 1.0 2.0 5.0 10.0 20.0 50.0 100.0)

llm_models=("BERT" "LLAMA3" "GPT2" "Doc2Vec")

datasets=(
  "./data/Algriculture|US_RetailBroilerComposite_Month.csv|monthly"
  "./data/Environment|NewYork_AQI_Day.csv|daily"
  "./data/Climate|US_precipitation_month.csv|monthly"
  "./data/Economy|US_TradeBalance_Month.csv|monthly"
  "./data/Energy|US_GasolinePrice_Week.csv|weekly"
  "./data/Public_Health|US_FLURATIO_Week.csv|weekly"
  "./data/Security|US_FEMAGrant_Month.csv|monthly"
  "./data/SocialGood|Unadj_UnemploymentRate_ALL_processed.csv|monthly"
  "./data/Traffic|US_VMT_Month.csv|monthly"
)


seeds=(2021)

get_forecast_settings() {
  local freq=$1
  case $freq in
    daily) echo "48 96 192 336|96|48" ;;
    weekly) echo "12 24 36 48|36|18" ;;
    monthly) echo "6 8 10 12|8|4" ;;
    *) echo "12 24 36 48|36|18" ;;
  esac
}

for llm_model in "${llm_models[@]}"; do
  llm_lower=${llm_model,,}  # lowercase
  for seed in "${seeds[@]}"; do
    for lr_multiplier in "${lr_multipliers[@]}"; do
      for model_name in "${models[@]}"; do
        for dataset_info in "${datasets[@]}"; do
          IFS='|' read -r root_path data_path frequency <<< "$dataset_info"
          model_id=$(basename ${root_path})

          settings=$(get_forecast_settings $frequency)
          IFS='|' read -r pred_lengths_str seq_len label_len <<< "$settings"
          pred_lengths=($pred_lengths_str)

          for pred_len in "${pred_lengths[@]}"; do
            for injection_mode in "${injection_modes[@]}"; do
              dataset_name=$(basename ${data_path} .csv)
              save_name="result_${model_name}_${dataset_name}_${llm_lower}_${injection_mode}"

              echo "Running $model_name on $dataset_name freq=$frequency pred_len=$pred_len mode=$injection_mode lr_mult=$lr_multiplier llm=$llm_lower"

              python -u run.py \
                --task_name long_term_forecast \
                --is_training 1 \
                --root_path $root_path \
                --data_path $data_path \
                --model_id ${model_id}_${seed}_${seq_len}_${pred_len}_fullLLM_0_${injection_mode}_${llm_lower} \
                --model $model_name \
                --data custom \
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
