#!/bin/bash
#
python iadbdev.py /home/data -r /home/weights  -b 64 -i 64 -e 100 -n iadb_64_p -s 1000 -o Perceptual # perceptual loss
# python iadbdev.py /home/data -r /home/weights  -b 64 -i 64 -e 100 -n iadb_64_l2 -s 1000 # L2 loss
