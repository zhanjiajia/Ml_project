"""
GroupID: 19
Topic: Identify plant species from herbarium specimens 
Student Name: Zhan Jia
Student ID: 20733225
Assignment #: Programming of Project
Student Email:19074569r@connect.polyu.hk
Course Name: COMP5212(L1)

"""

import numpy as np, pandas as pd, tensorflow as tf
import time
import os
import json
import codecs
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout

batchsize=256
learning_rate=0.001
epochs=20
shape = (200, 136, 3) 
debug=True
augment_data = False

start_time = time.time()
##############################Loading data###################################
# with codecs.open('D:/Python_Project/ML_Project/herbarium-2020-fgvc7/nybg2020/train/metadata.json', 'r', encoding='utf-8', errors='ignore') as f:
#     train_meta = json.load(f)
#
# with codecs.open('D:/Python_Project/ML_Project/herbarium-2020-fgvc7/nybg2020/test/metadata.json', 'r', encoding='utf-8', errors='ignore') as f:
#     test_meta = json.load(f)
#
# ###############Process training data###############################
# train_annotations = pd.DataFrame(train_meta['annotations'])
# train_categories = pd.DataFrame(train_meta['categories'])
# train_categories.columns = ['family', 'genus', 'category_id', 'category_name']
# train_images = pd.DataFrame(train_meta['images'])
# train_images.columns = ['file_name', 'height', 'image_id', 'license', 'width']
# train_regions = pd.DataFrame(train_meta['regions'])
# train_regions.columns = ['region_id', 'region_name']
# ############### Merge all the Dataframes################################
# train_data = train_annotations.merge(train_categories, on='category_id', how='outer')
# train_data = train_data.merge(train_images, on='image_id', how='outer')
# train_data = train_data.merge(train_regions, on='region_id', how='outer')
# ################Remove the rows with NaN values########################
# non = train_data.file_name.isna()
# kept = [x for x in range(train_data.shape[0]) if not non[x]]
# train_data = train_data.iloc[kept]
# dtypes = ['int32', 'int32', 'int32', 'int32', 'object', 'object', 'object', 'object', 'int32', 'int32', 'int32',
#           'object']
# for n, col in enumerate(train_data.columns):
#     train_data[col] = train_data[col].astype(dtypes[n])
#
# ###############Process testing data###############################
# test_images = pd.DataFrame(test_meta['images'])
# test_images.columns = ['file_name', 'height', 'image_id', 'license', 'width']
# #########################Save dataframe as a *.csv file########################
# train_data.to_csv('train_data.csv', index=False)
# test_images.to_csv('test_data.csv', index=False)
#
#
# #############Transform the family and genus to ids#########
# train_m = train_data[['file_name', 'family', 'genus', 'category_id']]
# train_fami = train_m.family.unique().tolist()
# train_m.family = train_m.family.map(lambda x: train_fami.index(x))
# train_genus = train_m.genus.unique().tolist()
# train_m.genus = train_m.genus.map(lambda x: train_genus.index(x))
# train_m.to_csv('train_data_trans.csv', index=False)

###############Data sets###############
# IRIS_TRAINING = "iris_training.csv"
# IRIS_TEST = "iris_test.csv"
# train_data = np.genfromtxt(IRIS_TRAINING, skip_header=None,
#                            dtype=float, delimiter=',')
# test_data = np.genfromtxt(IRIS_TEST, skip_header=None,
#                           dtype=float, delimiter=',')
# def get_data():
#     # Load datasets.
#     train_gen_data = ImageDataGenerator(featurewise_center=False,
#                                        featurewise_std_normalization=False,
#                                        rotation_range=180,
#                                        width_shift_range=0.1,
#                                        height_shift_range=0.1,
#                                        zoom_range=0.2)
#
#     return train_x, train_y, test_x, test_y
#
# ##########Trandform the famliy and genus to ids#########
# train_m = train_data[['file_name', 'family', 'genus', 'category_id']]
# train_fam_list = train_m.family.unique().tolist()
# train_m.family = train_m.family.map(lambda x: train_fam_list.index(x))
# train_genus_list = train_m.genus.unique().tolist()
# train_m.genus = train_m.genus.map(lambda x:train_genus_list.index(x))
#########################load data#####################
train_m_csv_file = "/content/drive/My Drive/My Drive/Learning/Machine learning project/train_data_trans_500.csv"
train_m_csv_data = pd.read_csv(train_m_csv_file, low_memory = False)
#train_m_total = pd.DataFrame(train_m_csv_data)
train_m = train_m_csv_data.iloc[:100]
#print(train_m)

test_cvs_file = "/content/drive/My Drive/My Drive/Learning/Machine learning project/test_data_short_real.csv"
test_csv_data = pd.read_csv(test_cvs_file, low_memory = False)
#test_data_total = pd.DataFrame(test_csv_data)
test_data = test_csv_data.iloc[:40]
#print(test_data)

dim_cate = train_m['category_id'].max() +1 ##the output dimension of the 'category_id' layer
dim_genu = train_m['genus'].max() +1    ##the output dimension of the 'genus_id' layer
dim_fami = train_m['family'].max() +1    ##the output dimension of the 'family_id' layer

####################Augmentation Data###########
train_x, train_y = train_test_split(train_m, test_size=0.05, shuffle=True, random_state=13)

data_gen1 = ImageDataGenerator(dtype='uint8')
data_gen2 = ImageDataGenerator(rotation_range=35, featurewise_center=False,
                           featurewise_std_normalization=False,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1, horizontal_flip=True,
                           dtype='uint8')


def crop(batch_x):
    cut1 = int(0.1 * batch_x.shape[1])
    cut2 = int(0.05 * batch_x.shape[2])
    return batch_x[:, cut1:-cut1, cut2:-cut2]


def crop_generator(batches, test=False):
    while True:
        if test:
            batch_x = next(batches)
            yield next(data_gen2.flow(crop(batch_x), batch_size=batchsize))
            #yield next(data_gen2.flow(batch_x, batch_size=batchsize))
        else:
            batch_x, batch_y = next(batches)
            yield (next(data_gen2.flow(crop(batch_x), batch_size=batchsize)), batch_y)
            #yield (next(data_gen2.flow(batch_x, batch_size=batchsize)), batch_y)


#i = 0
#for x, y in crop_generator(data_gen1.flow_from_dataframe(
#        dataframe=train_m, directory='/content/drive/My Drive/My Drive/Learning/Machine learning project/herbarium-2020-fgvc7/nybg2020/train/',
#        x_col="file_name", y_col=['family', 'genus', 'category_id'], class_mode="multi_output",
#        target_size=(shape[0], shape[1]), batch_size=batchsize,
#        validate_filenames=False, verbose=False)):
#    plt.imshow(x.astype('uint8')[1])
#    i = i + 1
#    if i == 1:
#        break
#################Defining model#################
def create_model():
    actual_shape = (crop(np.zeros((1, shape[0], shape[1], shape[2]))).shape)[1:]

    xi = Input(actual_shape)
    x = ResNet50(weights='imagenet', include_top=False, input_shape=actual_shape, pooling='max')(xi)
    '''
    xi = Input(shape)
    x =  Conv2D(8, (3, 3), strides=1, activation='relu', padding='same')(xi)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(16, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    '''
    x = Flatten()(x)
    y1 = Dense(dim_fami, name="family", activation='softmax')(x)
    y2 = concatenate([x, y1])
    y2 = Dense(dim_genu, name="genus", activation='softmax')(y2)
    y3 = concatenate([x, y1, y2])
    y3 = Dense(dim_cate, name="category_id", activation='softmax')(y3)
    model = Model(inputs=xi, outputs=[y1, y2, y3])
    model.layers[1].trainable = False
    model.get_layer('genus').trainable = False
    model.get_layer('category_id').trainable = False
    return model


def compile(model, learning_rate=0.005):
    model.compile(optimizer=Adam(learning_rate=0.005), loss=["sparse_categorical_crossentropy",
                                                             "sparse_categorical_crossentropy",
                                                             "sparse_categorical_crossentropy"],
                  metrics=['accuracy'])


train_steps = (train_x.shape[0] // batchsize) + 1
verify_setps = (train_y.shape[0] // batchsize) + 1
def train(epo, initial_epoch=0):
    return model.fit_generator(data_gen1.flow_from_dataframe(
                                    dataframe=train_x, directory='/content/drive/My Drive/My Drive/Learning/Machine learning project/herbarium-2020-fgvc7/nybg2020/train/',
                                    x_col="file_name", y_col=['family','genus','category_id'], class_mode="multi_output",
                                    target_size=(shape[0],shape[1]),batch_size=batchsize,
                                    validate_filenames=False, verbose=False),
					                    validation_data=data_gen1.flow_from_dataframe(dataframe=train_y, directory='/content/drive/My Drive/My Drive/Learning/Machine learning project/herbarium-2020-fgvc7/nybg2020/train/',
					                                    x_col="file_name", y_col=['family','genus','category_id'], class_mode="multi_output",
					                                    target_size=(shape[0],shape[1]),batch_size=batchsize,
					                                    validate_filenames=False, verbose=False),
						                  epochs=epo+initial_epoch,max_queue_size=30, workers=16, 
					                    initial_epoch=initial_epoch,
					                    steps_per_epoch=train_steps,
					                    validation_steps=verify_setps)
					                    
model = create_model()
compile(model,learning_rate)
model.summary()

############Train#############
for i in range(epochs):
    hist = train(1,i)
    genus_accu = hist.history['genus_accuracy'][0]
    fami_accu = hist.history['family_accuracy'][0]
    if fami_accu > 0.75:
        model.get_layer("family").trainable=False
        print("Stopped training family.")
        compile(model,learning_rate)
    else:
        model.get_layer("family").trainable=True
        print("Start again training family.")
        compile(model,learning_rate)

    if fami_accu > 0.5:
        model.get_layer("genus").trainable=True
        print("Training genus now.")
        compile(model,learning_rate)

    if genus_accu > 0.75:
        model.get_layer("genus").trainable=False
        print("Stopped training genus.")
        compile(model,learning_rate)
    else:
        model.get_layer("genus").trainable=True
        print("Start again training genus.")
        compile(model,learning_rate)
    if (genus_accu > 0.6) and (fami_accu > 0.6) :
        model.get_layer("category_id").trainable=True
        print("Training category now.")
        compile(model,learning_rate)
        
filename="weights.h5"
model.save_weights(filename)
print("Weights saved to {}".format(filename))


############Predict#######################

steps_predict = (test_data.shape[0]//batchsize)+1
test_data_gen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False)
predictions = model.predict_generator(test_data_gen.flow_from_dataframe(
                                dataframe=test_data, directory='/content/drive/My Drive/My Drive/Learning/Machine learning project/herbarium-2020-fgvc7/nybg2020/train/',
                                x_col="file_name",class_mode=None,
                                target_size=(shape[0],shape[1]),batch_size=batchsize,
                                validate_filenames=False, verbose=True),
                                 steps=steps_predict, workers=8)
'''                                 
predictions = model.predict_generator(crop_generator(data_gen1.flow_from_dataframe(
                                dataframe=test_data, directory='/content/drive/My Drive/My Drive/Learning/Machine learning project/herbarium-2020-fgvc7/nybg2020/train/',
                                x_col="file_name",class_mode=None,
                                target_size=(shape[0],shape[1]),batch_size=batchsize,
                                validate_filenames=False, verbose=True),True),
                                 steps=steps_predict, workers=8)'''
sub = pd.DataFrame()
sub['Real_category_id'] = test_data['category_id']
sub['Predicted_category_id'] = np.concatenate([np.argmax(predictions[2], axis=1), np.ones((len(test_data.category_id)-len(predictions[2])))], axis=0)
sub.to_csv('sub_category_id_resnet50.csv', index=False)
#sub.to_csv('sub_category_id.csv', index=False)

sub_fami = pd.DataFrame()
sub_fami['Real_family'] = test_data['family']
sub_fami['Predicted_family'] = np.concatenate([np.argmax(predictions[2], axis=1), np.ones((len(test_data.family)-len(predictions[2])))], axis=0)
sub_fami.to_csv('sub_family_resnet50.csv', index=False)
#sub_fami.to_csv('sub_family.csv', index=False)

sub_genus = pd.DataFrame()
sub_genus['Real_genus'] = test_data['genus']
sub_genus['Predicted_genus'] = np.concatenate([np.argmax(predictions[1], axis=1), np.ones((len(test_data.genus)-len(predictions[1])))], axis=0)
sub_genus.to_csv('sub_genus_resnet50.csv', index=False)
#sub_genus.to_csv('sub_genus.csv', index=False)
print("Submission file written. Total time elapsed: {} minutes".format((time.time()-start_time)//60))