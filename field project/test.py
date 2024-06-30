import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import random
import nibabel as nib
import pickle
import tensorlayer as tl

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import cv2
from sklearn.model_selection import train_test_split

X_train_input = []
X_train_target = []
'''
LGG_data_path = "data/MICCAI_BraTS_2018_Data_Training/LGG/Brats18_2013_24_1"
data_types = ['flair', 't1', 't1ce', 't2']
LGG_path_list = [x[0] for x in os.walk(LGG_data_path)]  #['LGG']
LGG_name_list = [os.path.basename(p) for p in LGG_path_list]
print(LGG_name_list)
index_LGG = list(range(0, len(LGG_name_list)))
random.shuffle(index_LGG)
tr_index_LGG = index_LGG[:-30]
LGG_name_train = [LGG_name_list[i] for i in tr_index_LGG]
print(LGG_name_train)
'''
data_types = ['flair', 't1', 't1ce', 't2']
LGG_data_path = 'data/MICCAI_BraTS_2018_Data_Training/LGG'
LGG_name_train = ['Brats18_2013_24_1']
global start_index
start_index = 0

def distort_imgs(data):
    """ data augmentation """
    #print(data.shape)
    x1, x2, x3, x4, y = data
    x1, x2, x3, x4, y = tl.prepro.flip_axis_multi([x1, x2, x3, x4, y], axis=1, is_random=True)  # left right
    x1, x2, x3, x4, y = tl.prepro.elastic_transform_multi([x1, x2, x3, x4, y], alpha=720, sigma=24, is_random=True)
    x1, x2, x3, x4, y = tl.prepro.rotation_multi([x1, x2, x3, x4, y], rg=20, is_random=True, fill_mode='constant')  # nearest, constant
    x1, x2, x3, x4, y = tl.prepro.shift_multi([x1, x2, x3, x4, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = tl.prepro.shear_multi([x1, x2, x3, x4, y], 0.05, is_random=True, fill_mode='constant')
    #x1, x2, x3, x4, y = tl.prepro.zoom_multi([x1, x2, x3, x4, y])
    return x1, x2, x3, x4, y

def vis_imgs(X, y, path):
    """ show one slice """
    global start_index
    if y.ndim == 2:
        y = y[:, :, np.newaxis]
    assert X.ndim == 3
    tl.vis.save_images(np.asarray([X[:, :, 0, np.newaxis]]),size=(1, 1),image_path='dataset/flair/'+str(start_index)+'.png')
    tl.vis.save_images(np.asarray([X[:, :, 1, np.newaxis]]),size=(1, 1),image_path='dataset/t1/'+str(start_index)+'.png')
    tl.vis.save_images(np.asarray([X[:, :, 2, np.newaxis]]),size=(1, 1),image_path='dataset/t1ce/'+str(start_index)+'.png')
    tl.vis.save_images(np.asarray([X[:, :, 3, np.newaxis]]),size=(1, 1),image_path='dataset/t2/'+str(start_index)+'.png')
    tl.vis.save_images(np.asarray([y]),size=(1, 1),image_path='dataset/label/'+str(start_index)+'.png')
    
    '''
    tl.vis.save_images(np.asarray([X[:, :, 0, np.newaxis],
                                   X[:, :, 1, np.newaxis], X[:, :, 2, np.newaxis],
                                   X[:, :, 3, np.newaxis], y]), size=(1, 5),
                       image_path=path)
    '''                       




def show_image_sample(X, y, task):
    global start_index
    # show one slice

    # print(X.shape, X.min(), X.max()) # (240, 240, 4) -0.380588 2.62761
    # print(y.shape, y.min(), y.max()) # (240, 240, 1) 0 1
    #vis_imgs(X, y, 'samples/{}/_train_im.png'.format(task))
    # show data augmentation results
    for i in range(10):
        x_flair, x_t1, x_t1ce, x_t2, label = distort_imgs([X[:, :, 0, np.newaxis], X[:, :, 1, np.newaxis],
                                                           X[:, :, 2, np.newaxis], X[:, :, 3, np.newaxis],
                                                           y])
        X_dis = np.concatenate((x_flair, x_t1, x_t1ce, x_t2), axis=2)
        vis_imgs(X_dis, label, 'samples/{}/_train_im_aug{}.png'.format(task, i))
    start_index = start_index + 1    

def image_normalization():
    data_types_mean_std_dict = {i: {'mean': 0.0, 'std': 1.0} for i in data_types}
    data_temp_list = []
    for i in data_types:
        data_temp_list = []
        for j in LGG_name_train:
            img_path = os.path.join(LGG_data_path, j, j + '_' + i + '.nii.gz')
            img = nib.load(img_path).get_data()
            data_temp_list.append(img)

        data_temp_list = np.asarray(data_temp_list)
        m = np.mean(data_temp_list)
        s = np.std(data_temp_list)
        data_types_mean_std_dict[i]['mean'] = m
        data_types_mean_std_dict[i]['std'] = s
    del data_temp_list
    print(data_types_mean_std_dict)
    with open('mean_std_dict.pickle', 'wb') as f:
        pickle.dump(data_types_mean_std_dict, f, protocol=4)
    return data_types_mean_std_dict

data_types_mean_std_dict = image_normalization()    

for i in LGG_name_train:
    all_3d_data = []
    for j in data_types:
        img_path = os.path.join(LGG_data_path, i, i + '_' + j + '.nii.gz')
        print(img_path)
        img = nib.load(img_path).get_data()
        img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
        img = img.astype(np.float32)
        all_3d_data.append(img)

    seg_path = os.path.join(LGG_data_path, i, i + '_seg.nii.gz')
    seg_img = nib.load(seg_path).get_data()
    seg_img = np.transpose(seg_img, (1, 0, 2))
    for j in range(all_3d_data[0].shape[2]):
        combined_array = np.stack((all_3d_data[0][:, :, j], all_3d_data[1][:, :, j], all_3d_data[2][:, :, j], all_3d_data[3][:, :, j]),axis=2)
        combined_array = np.transpose(combined_array, (1, 0, 2))
        combined_array.astype(np.float32)
        X_train_input.append(combined_array)
        seg_2d = seg_img[:, :, j]
        seg_2d.astype(int)
        X_train_target.append(seg_2d)
    del all_3d_data
    print("finished {}".format(i))

X_train_input = np.asarray(X_train_input, dtype=np.float32)
X_train_target = np.asarray(X_train_target)
y_train = X_train_target[:, :, :, np.newaxis]
y_train = (y_train > 0).astype(int)
print(str(X_train_input.shape)+" "+str(y_train.shape))

#X = np.asarray(X_train_input[80])
#y = np.asarray(y_train[80])
#print(str(X.shape)+" "+str(y.shape))

def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def getModel(input_size=(240,240,4)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs=[inputs], outputs=[conv10])



dim = 240
X_train, X_test, y_train, y_test = train_test_split(X_train_input, y_train, test_size = 0.10, random_state = 1)
#X_train = X_train.reshape(len(X_train),dim,dim,1)
#y_train = y_train.reshape(len(y_train),dim,dim,1)
#X_test = X_test.reshape(len(X_test),dim,dim,1)
#y_test = y_test.reshape(len(y_test),dim,dim,1)
images = np.concatenate((X_train,X_test),axis=0)
mask  = np.concatenate((y_train,y_test),axis=0)
'''
tr = X_train[12]
yr = y_train[12]

cv2.imshow('tr',tr)
cv2.imshow('yr',yr)
cv2.waitKey(0)
'''


model = getModel(input_size=(240,240,4))
'''
model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])
print(model.summary())
model.compile(optimizer=Adam(lr=2e-4), loss=[dice_coef_loss], metrics = [dice_coef, 'binary_accuracy'])

train_vol, validation_vol, train_seg, validation_seg = train_test_split((images-127.0)/127.0, 
                                                            (mask>127).astype(np.float32), 
                                                            test_size = 0.1,random_state = 2018)

train_vol, test_vol, train_seg, test_seg = train_test_split(train_vol,train_seg, 
                                                            test_size = 0.1, 
                                                            random_state = 2018)

hist = model.fit(x = train_vol, y = train_seg, batch_size = 16, epochs = 10, validation_data =(test_vol,test_seg))
model.save_weights('model/model_weights1.h5')            
model_json = model.to_json()
with open("model/model1.json", "w") as json_file:
    json_file.write(model_json)
f = open('model/history1.pckl', 'wb')
pickle.dump(hist.history, f)
f.close()

'''
with open('model/model1.json', "r") as json_file:
    loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)

model.load_weights("model/model_weights1.h5")
model._make_predict_function()   
#print(model.summary())
temp = []
temp.append(X_train_input[10])
temp = np.asarray(temp)
temp = (temp-127.0)/127.0
print(temp.shape)
preds = model.predict(temp)
preds = preds[0]
print(preds.shape)
for i in range(0,X_train_input.shape[0]):
    show_image_sample(X_train_input[i],y_train[i],"all")
#cv2.imshow("ll",cv2.resize(preds*255,(400,400),interpolation = cv2.INTER_CUBIC))
#cv2.waitKey(0)


