#!/bin/bash

python -m uni2ts.data.builder.simple \
  A \
  "data/csv/A.csv" \
  --dataset_type "wide_multivariate" \
  --freq "5min"
