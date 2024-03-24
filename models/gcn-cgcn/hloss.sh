# python train.py --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --data_dir ./../dataset/tacred/json --vocab_dir ./../dataset/tacred/vocab --id hl-bs64 --hier_dist --batch_size 64

#LCA
python train.py --seed 0 --prune_k 1 --lr 0.3 --rnn_hidden 200 --num_epoch 100 --pooling max --mlp_layers 2 --pooling_l2 0.003 --data_dir ./../dataset/tacred/json --vocab_dir ./../dataset/tacred/vocab --id hl-a0p5 --hier_dist --batch_size 64