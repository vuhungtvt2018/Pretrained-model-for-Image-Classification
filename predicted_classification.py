import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import glob
import cv2
import random
import argparse
import os

def resize(im, a, height_new_size=250 , width_new_size=180):   
    desired_size = a[0]
    old_size = im.shape[:2]
    ratio = float(desired_size)/max(old_size)
    new_size = (height_new_size , width_new_size)#(250, 180)

    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    k = min(random.randint(100, 280), 255)
    color = [100, 100, 100]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im


# class RandomCutout(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.mask_size = (20, 20)
#     def call(self, image, training=True):
#         def augment_image():
#             return tfa.image.random_cutout(image, mask_size = self.mask_size, constant_values = (100, 100, 100))
#         training = tf.constant(training, dtype=tf.bool)

#         rotated = tf.cond(training, augment_image, lambda: image)
#         rotated.set_shape(rotated.shape)
#         return rotated


def predict(imgs=250,height_new_size=250 , width_new_size=180,model_name="aaa.h5"):
    for i in glob.glob("dataset/*"):
        image=cv2.imread(i)
        image=resize(image,(imgs,imgs),height_new_size=height_new_size , width_new_size=width_new_size)
        cv2.imwrite(os.path.join("predict",os.path.basename(i)),image)
    
    new_model = tf.keras.models.load_model(os.path.join("model",model_name),compile=False)#,  custom_objects={'RandomCutout': RandomCutout()})
    # Show the model architecture
    new_model.summary()

    IMAGE_FEED=[]
    name=[]
    img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    for i in glob.glob('predict/*'):
        print(i)
        if i.split('.')[-1].lower() in img_formats:
                name.append(i)
                image=cv2.imread(i)
                IMAGE_FEED.append(image)

    image = np.stack(IMAGE_FEED, axis = 0)
    predictions1 = new_model.predict_on_batch(image)


    test_real_dataset = tf.keras.utils.image_dataset_from_directory('classify/image/test',
                                                                    shuffle=True,
                                                                    batch_size=1,
                                                                    image_size=(imgs,imgs))
    print(*test_real_dataset.class_names, sep=", ")



    np.max(predictions1, axis=1)
    list_max=np.argmax(predictions1, axis=1).tolist()
    np.take(np.array(test_real_dataset.class_names), list_max)
    print(*list_max, sep=", ")


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs', type=int, default=250,help="kích thước ảnh output")
    parser.add_argument('--height_new_size', type=int, default=250,help="kích thước ảnh trước khi resize về size ảnh imgs (trước khi thêm padding để resize về 250)")
    parser.add_argument('--width_new_size', type=int, default=180,help="kích thước ảnh trước khi resize về size ảnh imgs (trước khi thêm padding để resize về 250)")
    parser.add_argument('--model_name', type=str, default="aaa.h5",help="ten model")
    
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    predict(opt.imgs, opt.height_new_size, opt.width_new_size,opt.model_name) 

# python augmentation.py --path_to_data C:\Users\Admin\Desktop\dau_bua\data\classify --path_des C:\Users\Admin\Desktop\dau_bua\data\classify\image\train --class_name NG OK --num_image 3500 --probability 0.5 --imgs 220 --height_new_size 220 --width_new_size 200 --PATH C:\Users\Admin\Desktop\dau_bua\data\classify\image --batch_size 64 --epoch 20 --three_rd_training --name_model abc