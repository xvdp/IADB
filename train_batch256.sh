#!/bin/bash
# run(data_folder, results, batch_size, image_size, epochs, lr_rate, save_every)
# mod of iadb with batch size 64, requires 48GB GPU
# 
# 634 (256 sized) iterations/epoch * 100 epochs -> 23.4 hris
python iadb.py /home/data -r /home/weights -b 256 -i 64 -e 100 -n iadb_256

# to continue  training
# python iadb.py /home/data -r /home/weights -b 256 -i 64 -e 100 -n iadb_256 -c celeba_256_00063000.ckpt
