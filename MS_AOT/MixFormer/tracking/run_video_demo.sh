# We only support manually setting the bounding box of first frame and save the results in debug directory.

##########-------------- MixFormer-22k-----------------##########
python tracking/video_demo.py mixformer_online baseline /home/yoxu/vot_2022/visualized_tracker/vedio/ball2.avi  \
  --optional_box 529 48 19 17 --params__model mixformer_online_22k.pth.tar --debug 1 \
  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3


##########-------------- MixFormerL-22k-----------------##########
#python tracking/video_demo.py mixformer_online baseline /home/cyt/project/MixFormer/test.mp4  \
#   --optional_box 408 240 94 254 --params__model mixformerL_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

