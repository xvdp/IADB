#!/bin/bash
# train(data_folder, results, batch_size, image_size, epochs, lr_rate, save_every)
python iadb.py /home/data -r /home/weights -b 256 -i 64 -e 100 -n iadb_256
