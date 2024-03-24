# for SEED in 78 23 61;
# do python train_retacred.py --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base;
# done;
SEED=78
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --lca

# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --filter_out
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --filter_out
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --lca --filter_out

# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --filter_out --relabel
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --filter_out --relabel
# python train_retacred.py --data_dir ./../dataset/re-tacred --model_name_or_path bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --lca --filter_out --relabel
