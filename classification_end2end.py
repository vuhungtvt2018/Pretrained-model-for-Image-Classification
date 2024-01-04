import Augmentor
import os
import shutil
import random
import glob
import cv2
import argparse
import time
import tensorflow as tf

#AUGMENTATION
def move_files(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    files = os.listdir(src_folder)

    for file_name in files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.move(src_path, dest_path)
        print(f"Đã di chuyển: {file_name} từ {src_folder} đến {dest_folder}")


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


def pre_processing_image(path_to_data="classify",
                        path_des="classify\\image\\train",
                        class_name="NG",
                        num_image=1000,
                        probability=0.5,
                        max_left_rotation_rotate=5, max_right_rotation_rotate=5, fillcolor_rotate=100,
                        grid_width_distortion=4, grid_height_distortion=4, magnitude_distortion=4,
                        magnitude_skew=0.3,
                        min_contrast=1.5,max_contrast=3,
                        min_brightness=1.5,max_brightness=3,
                        max_shear_left=4,max_shear_right=4,
                        imgs=250,height_new_size=250 , width_new_size=180):
    
    probability_rotate=probability_distortion=probability_skew=probability_contrast=probability_brightness=probability_shear=probability
    
    path_to_data = os.path.join(path_to_data,class_name)
    path_des = os.path.join(path_des,class_name)
    print(path_to_data)
    print(path_des)
    
    if os.path.exists(path_des):
        shutil.rmtree(path_des)
        os.makedirs(path_des)
    else:
        os.makedirs(path_des)
    
    p = Augmentor.Pipeline(path_to_data)
    p.rotate_without_crop(probability=probability_rotate, max_left_rotation=max_left_rotation_rotate, max_right_rotation=max_right_rotation_rotate, expand=True, fillcolor=(fillcolor_rotate))
    p.random_distortion(probability=probability_distortion, grid_width=grid_width_distortion, grid_height=grid_height_distortion, magnitude=magnitude_distortion)
    p.skew(probability=probability_skew, magnitude=magnitude_skew)
    p.random_contrast(probability=probability_contrast, min_factor=min_contrast,max_factor=max_contrast)
    p.random_brightness(probability=probability_brightness,min_factor=min_brightness,max_factor=max_brightness)
    p.shear(max_shear_left=max_shear_left,max_shear_right=max_shear_right,probability=probability_shear)
    p.process()
    p.sample(num_image)
    
    for i in glob.glob(os.path.join(path_to_data,"output")+"\\*"):
        image=cv2.imread(i)
        image=resize(image,(imgs,imgs),height_new_size=height_new_size , width_new_size=width_new_size)
        cv2.imwrite(i,image)
    
    move_files(os.path.join(path_to_data,"output"), path_des)
##############################################################################################################################

#CLASSIFICATION
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def splt_train_val(path_train,path_test):
    lis=os.listdir(r"D:\HUNG\c\train")
    for j in lis:
        #if j=="D":
            source=os.path.join(path_train,j)
            dest=os.path.join(path_test,j)
            no_of_files=int(len(os.listdir(source))*0.15)
            print("%"*25+"{ Details Of Transfer }"+"%"*25)
            print("\n\nList of Files Moved to %s :-"%(dest))
            for i in range(no_of_files):
                random_file=random.choice(os.listdir(source))
                source_file="%s/%s"%(source,random_file)
                dest_file=dest
                shutil.move(source_file,dest_file)
            print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)
            print(i)
        
        
# class RandomCutout(tf.keras.layers.experimental.preprocessing.PreprocessingLayer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.mask_size = (50, 50)
#     def call(self, image, training=True):
#         def augment_image():
#             return tfa.image.random_cutout(image, mask_size = self.mask_size, constant_values = (100, 100, 100))
#         training = tf.constant(training, dtype=tf.bool)

#         rotated = tf.cond(training, augment_image, lambda: image)
#         rotated.set_shape(rotated.shape)
#         return rotated


def classify(PATH="classify/image",batch_size=128,imgs=250,epoch=40,three_rd_training=False,name_model="abc"):  
    s=time.time()
    PATH = PATH# "D:/HUNG/b/image"
    
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'test')

    splt_train_val(train_dir,validation_dir)


    BATCH_SIZE = batch_size
    IMG_SIZE = (imgs, imgs)
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=256,
                                                                batch_size=BATCH_SIZE,
                                                                image_size=IMG_SIZE)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                    shuffle=True,
                                                                    batch_size=BATCH_SIZE,
                                                                    image_size=IMG_SIZE)
    
    class_names = train_dataset.class_names
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    print('Number of training batches: %d' % tf.data.experimental.cardinality(train_dataset))
    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)



    data_augmentation = tf.keras.Sequential([
                                            tf.keras.layers.RandomRotation(0.02),
                                            tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
                                            tf.keras.layers.RandomBrightness((-0.2, 0.2)),
                                            tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0.2)),
                                            tf.keras.layers.GaussianNoise(0)])
                                            #RandomCutout()])


    #preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model =tf.keras.applications.resnet_v2.ResNet50V2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    print(base_model.summary())
    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_names), activation = 'softmax')
    inputs = tf.keras.Input(shape=(imgs, imgs, 3))
    y = data_augmentation(inputs)
    y = preprocess_input(y)
    y = base_model(y, training=False)
    y = global_average_layer(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Dense(128, activation = 'relu')(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    outputs = prediction_layer(y)
    model= tf.keras.Model(inputs, outputs)
    print(model.summary())
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    initial_epochs = 10

    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=validation_dataset)
    time.sleep(5)
    fine_tune_at = 150
    base_model.trainable = True
    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:150]:
        layer.trainable = False
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    base_learning_rate = 0.0001
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),  metrics=['accuracy'])
    history_fine = model.fit(train_dataset,
                            epochs=epoch,
                            validation_data=validation_dataset)#, callbacks=[callback])
    
    model.save('model/1'+name_model+'.h5')
    
    time.sleep(5)
    if three_rd_training:
        fine_tune_at = 150
        base_model.trainable = True
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:]:
            layer.trainable = True
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        base_learning_rate = 0.0001
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/20),  metrics=['accuracy'])
        history_fine = model.fit(train_dataset,
                                epochs=epoch,
                                validation_data=validation_dataset, callbacks=[callback])

        model.save('model/2'+name_model+'.h5')
    
    print("thời gian training ", (time.time()-s)//60)
    

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_data', type=str, default="classify",help="thu mục lưu gốc NG/OK")
    parser.add_argument('--path_des', type=str, default="classify/image/train")
    parser.add_argument('--class_name', type=str, nargs='+', default=["NG","OK"],help="tên class cần augmentation")
    parser.add_argument('--num_image', type=int, default=1000,help="số lượng ảnh")
    parser.add_argument('--probability', type=float, default=0.5)
    
    #PHÉP XOAY ẢNH
    parser.add_argument('--max_left_rotation_rotate', type=int, default=5)
    parser.add_argument('--max_right_rotation_rotate', type=int, default=5)
    parser.add_argument('--fillcolor_rotate', type=int, default=100)
    
    #PHÉP distortion
    parser.add_argument('--grid_width_distortion', type=int, default=5)
    parser.add_argument('--grid_height_distortion', type=int, default=5)
    parser.add_argument('--magnitude_distortion', type=int, default=4)
    
    #PHÉP skew
    parser.add_argument('--magnitude_skew', type=float, default=0.3)
    
    #CONTRAST
    parser.add_argument('--min_contrast', type=float, default=1.5)
    parser.add_argument('--max_contrast', type=float, default=3)
    
    #BRIGHTNESS
    parser.add_argument('--min_brightness', type=float, default=1.5)
    parser.add_argument('--max_brightness', type=float, default=3)
    
    #SHEAR
    parser.add_argument('--max_shear_left', type=int, default=4)
    parser.add_argument('--max_shear_right', type=int, default=4)
    
    #THÔNG SỐ DATA
    parser.add_argument('--imgs', type=int, default=250,help="kích thước ảnh output")
    parser.add_argument('--height_new_size', type=int, default=250,help="kích thước ảnh trước khi resize về size ảnh imgs (trước khi thêm padding để resize về 250)")
    parser.add_argument('--width_new_size', type=int, default=180,help="kích thước ảnh trước khi resize về size ảnh imgs (trước khi thêm padding để resize về 250)")
    
    #THÔNG SỐ TRAINING
    parser.add_argument('--PATH', type=str, default="classify/image",help="thu mục data training")
    parser.add_argument('--batch_size', type=int, default=32,help="batch size")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--three_rd_training', action='store_true', help='Enable 3rd_training')
    parser.add_argument('--name_model', type=str, default="abc",help="thu mục weight")
    
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

if __name__ == '__main__':
    opt = parse_opt()
    print(opt)
    for i in opt.class_name:
        pre_processing_image(opt.path_to_data, opt.path_des, i, opt.num_image, opt.probability,\
                            opt.max_left_rotation_rotate, opt.max_right_rotation_rotate, opt.fillcolor_rotate,\
                            opt.grid_width_distortion, opt.grid_height_distortion, opt.magnitude_distortion,\
                            opt.magnitude_skew,\
                            opt.min_contrast, opt.max_contrast,\
                            opt.min_brightness, opt.max_brightness,\
                            opt.max_shear_left, opt.max_shear_right,\
                            opt.imgs, opt.height_new_size, opt.width_new_size)
    
    classify(opt.PATH, opt.batch_size, opt.imgs, opt.epoch, opt.three_rd_training, opt.name_model)