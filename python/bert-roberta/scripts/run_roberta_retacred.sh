#for SEED in 78 23 61;
#do python train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
#done;

SEED=78
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --lca

# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --filter_out
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --lca --filter_out

# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --relabel
python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --filter_out --relabel
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --lca --filter_out --relabel