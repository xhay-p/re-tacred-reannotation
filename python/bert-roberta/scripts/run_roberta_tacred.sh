# for SEED in 78 23 61;
# for SEED in 78;
# do python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
# done;

SEED=78
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --hier_loss --lca

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --hier_loss --lca

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --relabel
python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --relabel --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --filter_out --relabel --hier_loss --lca



#ALL POSITIVE CLASSES RELATION CLASSIFICATION
#--------------------------------------------
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --hier_loss --lca

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out --hier_loss --lca

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out --relabel
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out --relabel --hier_loss
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --all_pos --filter_out --relabel --hier_loss --lca