sudo apt-get install libturbojpeg0

# install libGL.so.1
sudo apt update
sudo apt install libgl1-mesa-glx

# install gcc&g++ 7.4.0
sudo apt-get update
sudo apt-get install gcc-7
sudo apt-get install g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
sudo update-alternatives --config g++

sudo apt-get install ninja-build

# copy our runtime environment to $HOME/anaconda3/envs/"
cp -r ms_aot $HOME/anaconda3/envs/


echo "Suppose that you have installed the anaconda3 at $HOME"
echo "If you have any question about the environment, please feel free to email 22151080@zju.edu.cn"

if [ ! -d "$HOME/anaconda3" ]; then
  echo "Please install anaconda3 at $HOME!!!"
  exit 1
fi

if [ ! -d "$HOME/anaconda3/envs/ms_aot" ]; then # if you want change the name of environment, you should modify this command
  echo "Please follow readme to create the right environment" 
  exit 1
fi

CUR_ROOT = `pwd` # the same as the vot workspace
if [ ! -f "$CUR_ROOT/MS_AOT/pretrain_models/ms_aot_model.path" ]; then
  echo "Please download ms_aot_model.path from:"
  echo "https://drive.google.com/file/d/1aEvMAcx3sJ2FRBIb0MsCSsvJrp3M0Cce/view?usp=sharing"
  echo "Then move it to $CUR_ROOT/MS_AOT/pretrain_models/ms_aot_model.path"
  exit 1
fi

if [ ! -f "$CUR_ROOT/MS_AOT/MixFormer/models/mixformerL_online_22k.pth.tar" ]; then
  echo "Please download mixformerL_online_22k.pth.tar from:"
  echo "https://drive.google.com/file/d/18qfUTVOyQ7Nyz8QaEoa2zecVbQCnbtWV/view?usp=sharing"
  echo "Then move it to $CUR_ROOT/MS_AOT/MixFormer/models/"
  exit 1
fi
# source $HOME/anaconda3/etc/profile.d/conda.sh
# conda activate ms_aot

cd vot_submit







