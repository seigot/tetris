#!/bin/bash

python start.py -l 1 -f test_l1_random.log
python start.py -l 2 -f test_l2_random.log
python start.py -l 3 -f test_l3_random.log
python start.py -l 1 -s y -f test_l1_sample.log
python start.py -l 2 -s y -f test_l2_sample.log
python start.py -l 3 -s y -f test_l3_sample.log

