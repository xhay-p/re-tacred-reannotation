#!/bin/sh
# for SEED in 78 23 61;
# for SEED in 78;
# do python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base;
# done;

SEED=78
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --filter_out
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --filter_out

# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --filter_out --relabel
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --filter_out --relabel



## TEST SAVE MODEL
# python train_tacred_save.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --output_dir bert-original
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --hier_normalise
# python train_tacred.py --data_dir ./../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --hier_loss --hier_log


python train_tacred_save.py --data_dir ./../../dataset/tacred/json --model_name_or_path ~/pretrained_bert/bert-base-cased --input_format typed_entity_marker --seed $SEED --train_batch_size 64 --test_batch_size 64 --learning_rate 5e-5 --gradient_accumulation_steps 1 --run_name bert-base --align_retacred --output_dir bert 