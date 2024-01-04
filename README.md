# Pretrain Classification with limited Dataset (with Tensorflow)
## Đã được sử dụng cho model OCV, nhận diện các kí tự và kiểm tra độ mờ, độ mất nét của kí tự trên sản phẩm thực tế
   - Image Classification sẽ được pretrain trên tập dữ liệu augmentation (được augment từ tập dữ liệu thực tế) mà không train trên tập dữ liệu thực. 
   - Model được đánh giá và test trên tập dữ liệu thực tế.
   - Custom RandomCutout layer.
   - Custom Dense layer.
   - Backbone: Resnet50v2. Có thể thay thế các backbone khác mà tensorflow cung cấp.
   - Convert model sang onnx.
   - Sử dụng chiến lược training từ top layer về bottom layer.
   - Trích xuất được các embedding vector. Dùng cho việc tính similarity distance.

## Install
   ```
   git clone https://github.com/vuhungtvt2018/pretrain_classification.git  # clone 
   cd pretrain_classification 
   pip install -r requirements.txt  # install 
   mkdir model  # thư mục chứa file weight sau khi training 
   mkdir predict # thư mục chứa ảnh test cho việc predict (Ảnh trong thư mục dataset sau khi pre_processing sẽ di chuyển vào đây)
   ```
## Requirements 
   python<=3.9.18 \
   tensorflow-gpu<=2.9.3 \
   keras<=2.9.0 \
   augmentor<=0.2.12
## Dataset
   Số lượng ảnh của mỗi class: tối thiểu 10 ảnh
## Pre-trained models
   Sử dụng pretrain Resnet50v2. \
   Có thể thay thế các pretrain khác mà tensorflow cung cấp.
## Training
   Thời gian training 40 phút
   ```
   python classification_end2end.py --path_to_data classify --path_des classify/image/train --class_name NG OK --num_image 3500 --probability 0.5 --max_left_rotation_rotate 5 
   --max_right_rotation_rotate 5 --fillcolor_rotate 100 --grid_width_distortion 5 --grid_height_distortion 5 --magnitude_distortion 4 --magnitude_skew 0.3 --min_contrast 1.5 
   --max_contrast 3 --min_brightness 1.5 --max_brightness 3 --max_shear_left 4 --max_shear_right 4 --imgs 220 --height_new_size 220 --width_new_size 200 --PATH classify/image 
   --batch_size 64 --epoch 20 --three_rd_training --name_model abc
   ```
## Testing
   ```
   python predicted_classification.py --imgs 220 --height_new_size 220 --width_new_size 200 --model_name 1abc.h5
   ```
