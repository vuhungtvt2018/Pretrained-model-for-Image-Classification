# Pretrain Classification with limited Dataset (tensorflow)
## Install
   '''
   git clone https://github.com/vuhungtvt2018/pretrain_classification.git  # clone \
   cd pretrain_classification \
   pip install -r requirements.txt  # install \
   mkdir model  # thư mục chứa file weight sau khi training \
   mkdir predict # thư mục chứa ảnh test cho việc predict (Ảnh trong thư mục dataset sau khi pre_processing sẽ di chuyển vào đây)
   '''
## Dataset
   Số lượng ảnh của mỗi class: 20 ảnh
## Pre-trained models
   Sử dụng pretrain Resnet50v2. Có thể thay thế các pretrain khác mà keras cung cấp
## Training
   Thời gian training 40 phút
   ```
   python classification_end2end.py --path_to_data classify --path_des classify/image/train --class_name NG OK --num_image 3500 --probability 0.5 --max_left_rotation_rotate 5 \
   --max_right_rotation_rotate 5 --fillcolor_rotate 100 --grid_width_distortion 5 --grid_height_distortion 5 --magnitude_distortion 4 --magnitude_skew 0.3 --min_contrast 1.5 --max_contrast 3 --min_brightness 1.5 --max_brightness 3 --max_shear_left 4 \
   --max_shear_right 4 --imgs 220 --height_new_size 220 --width_new_size 200 --PATH classify/image --batch_size 64 --epoch 20 --three_rd_training --name_model abc
   ```
## Testing
   ```
   python predicted_classification.py --imgs 220 --height_new_size 220 --width_new_size 200 --model_name 1abc.h5
   ```
