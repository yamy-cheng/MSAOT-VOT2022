#! /bin/bash

for i in {1..20}
do
    CUDA_VISIBLE_DEVICES=1 vot evaluate --workspace /home/cym/project/vot2022sts MS_AOT
done

vot analysis --workspace /home/cym/project/vot2022sts MS_AOT