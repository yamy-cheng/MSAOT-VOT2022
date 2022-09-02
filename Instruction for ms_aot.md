# Instruction
Hi Committee 

The singularity container that we build was uploaded to Singularity-Hub named ms_aot, The committee can easily pull the ms_aot container by ‘singularity pull --arch amd64 library://qiangming/ms_aot/ms_aot:latest’ command.

 Specifically, The tutorial of use singularity container to run ms_aot is follow.

## Hardware Requirements
* GPU: GeForce RTX 3090 (24 GB memory)
* CPU: Intel(R) Xeon(R) Silver 4210R CPU @ 2.40GHz 
* Memory: 190 GB
* HD: 879 GB

## Run MS-AOT

* Use `singularity pull` to build image that we made
```
singularity pull --arch amd64 library://qiangming/ms_aot/ms_aot:latest
```

* Run ms_aot_latest.sif
```
singularity shell --nv ms_aot_latest.sif
```

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

* To get results, use `vot-toolkit` 
```
vot evaluate --workspace $(pwd) MS_AOT
```
* or use our script
```
chmod +x evaluate_ms_aot.sh
./evaluate_ms_aot.sh
```

## **Note**: 
1. We recommend to use `evaluate_ms_aot.sh` to get results, since the vot toolkit will sometimes be interrupted due to the slow building of models. Before run `evaluate_ms_aot.sh`, you should modify the `workspace` defined in it.
2. We recommend to use GeForce RTX 3090 to run the program so that can avoid something unknown problems.

# Contact
If you have any problems about the environment settings and result reproduction, feel free to email 22151080@zju.edu.cn and we will reply as soon as possible ^_^