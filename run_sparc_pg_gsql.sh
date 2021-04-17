
LOGDIR="sparc_pg_gsql_logger_dir"

CUDA_VISIBLE_DEVICES=0 python3 run.py --raw_train_filename="data/sparc_data/train.pkl" \
          --raw_validation_filename="data/sparc_data/dev.pkl" \
          --database_schema_filename="data/sparc_data/tables.json" \
          --embedding_filename=$GLOVE_PATH \
          --data_directory="processed_data_sparc" \
          --input_key="utterance" \
          --state_positional_embeddings=1 \
          --use_query_attention=1 \
          --discourse_level_lstm=1 \
          --interaction_level=1 \
          --reweight_batch=1 \
          --freeze=1 \
          --train=1 \
          --logdir=$LOGDIR \
          --evaluate=1 \
          --evaluate_split="valid" \
          --use_predicted_queries=1 
