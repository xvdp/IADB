#!/bin/bash
# train(data_folder, results, batch_size, image_size, epochs, lr_rate, save_every)
# uses 12GB GPU - approx training time 
# 1.25 s/iteration * 2536 (64 sized) iterations/epoch * 100 epochs ~= 88 hrs
python iadb.py /home/data -r /home/weights  -b 64 -i 64 -e 100 -n iadb_64 -s 200
