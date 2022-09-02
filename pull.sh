#! /bin/bash

for i in {1..20}
do
    singularity push -U ms_aot_v2.sif library://qiangming/ms_aot/ms_aot:v2
done
echo "push done!!!"