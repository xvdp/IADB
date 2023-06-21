#!/bin/bash
# run(data_folder, results, batch_size, image_size, epochs, lr_rate, save_every, checkpoint, test=True)
# test pretrained - requires trained checkpoint
# 
python iadb.py /home/data -r /home/weights -b 256 -i 64 -e 100 -n iadb_256 -c celeba_256_00063000.ckpt -t 
#python iadb.py /home/data -r /home/weights -b 64 -i 64 -e 100 -n iadb_64 -c celeba_64_00254200.ckpt -t 
