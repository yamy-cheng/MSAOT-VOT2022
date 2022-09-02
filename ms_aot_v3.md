# Instruction
Hi Committee

We build a new singularity container named `ms_aot_v3` that include a conda environment. Running MS_AOT tracker in this new container, The problem you met may disappear. You can get the .def in [ms_aot_v3.def](https://drive.google.com/file/d/1MwL1CVs_Yc-jGjiS-4pBFVspm0DTIPh7/view?usp=sharing).


## Preparation of running
*  You need to build the container by this command.

```
singularity build --fakeroot ms_aot_v3.sif ms_aot_v3.def
```

* Run ms_aot_v3.sif

```
singularity shell --nv ms_aot_v3.sif
```

* Initialize the conda config and activate `ms_aot` conda environment
  
```
conda init
source /opt/conda/etc/profile.d/conda.sh
conda activate ms_aot
```

## Running MS_AOT tracker in container

* Run the following command to set paths for this MixFormer
  
```
cd MS_AOT/MixFormer
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
* return to the path of MS_AOT_submit 
```
cd -
```
* Download the checkpoint of [MS-AOT](https://drive.google.com/file/d/1aEvMAcx3sJ2FRBIb0MsCSsvJrp3M0Cce/view?usp=sharing), and move it into `MS_AOT/pretrain_models`:
```
mv ms_aot_model.pth MS_AOT/pretrain_models
```
* Download the checkpoint of [MixFormer](https://drive.google.com/file/d/18qfUTVOyQ7Nyz8QaEoa2zecVbQCnbtWV/view?usp=sharing), and move it into `MS_AOT/MixFormer/models`:
  
```
mv mixformerL_online_22k.pth.tar MS_AOT/MixFormer/models
```

* Remove the `config.yaml` and use vot-toolkit-python to initialize the workspace, 
```
rm config.yaml
vot initialize <stack-name> --workspace <workspace-path>
```

* Modify the `"paths"` and `"env_PATH"` in the `trackers.ini` regarding your environment

* To get results, use our script

```
chmod +x evaluate_ms_aot.sh
./evaluate_ms_aot.sh
```

## NOTE
1. We recommend to use evaluate_ms_aot.sh to get results, since the vot toolkit will sometimes be interrupted due to the slow building of models. Before run `evaluate_ms_aot.sh`, you should modify the `workspace` defined in it.
