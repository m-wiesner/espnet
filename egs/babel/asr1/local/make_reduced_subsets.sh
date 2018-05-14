#!/bin/bash

subset_factors="2 5 10 20 50 100"

. ./utils/parse_options.sh

if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/make_reduced_subsets.sh <train>"
  exit 1; 
fi

train=$1

train_dir=`dirname ${train}`
trainname=`basename ${train}`

num_utts=`cat $train/text | wc -l`
for i in ${subset_factors}; do
  echo "Factor: ${i}"
  num_utts_=$(( ${num_utts} / ${i} ))
  frac=`echo $i | awk '{print 1/$1}'`
  ./utils/subset_data_dir.sh $train ${num_utts_} ${train_dir}/${trainname}_${frac}
done

exit 0;
