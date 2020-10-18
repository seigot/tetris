#!/bin/bash

## default level
VALUE_L="1"

## get args level setting
while getopts l: OPT
do
  case $OPT in
    "l" ) VALUE_L="$OPTARG" ;;
  esac
done
echo "level: $VALUE_L"

## set parameter for each level
RANDOM_SEED="-1"

case $VALUE_L in
    "1" ) RANDOM_SEED="0" ;;
    "2" ) RANDOM_SEED="-1" ;;
    "3" ) RANDOM_SEED="-1" ;;
    * ) echo "invalid level: $VALUE_L"; exit 1;;
esac

## start game
python3 tetris_manager/tetris_game.py --seed ${RANDOM_SEED}
