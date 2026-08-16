import pandas as pd
import os
from shutil import copyfile

df_label_train = pd.read_csv('./Labels/CSV Format/train_labels.csv')
df_label_test = pd.read_csv('./Labels/CSV Format/test_labels.csv')

filename_train = df_label_train ['filename']
filename_test = df_label_test ['filename'] # val

#tạo các thư mục 
try:
    os.mkdir('./train_data')
    os.mkdir('./train_data/images')
    os.mkdir('./train_data/labels')
    os.mkdir('./train_data/images/train')
    os.mkdir('./train_data/images/val')
    os.mkdir('./train_data/labels/train')
    os.mkdir('./train_data/labels/val')
except OSError:
    pass

def split_train(SOURCE, DESTINATION_TRAIN,  DESTINATION_TEST):
    for filename in filename_train:
        this_file = SOURCE + filename
        destination = DESTINATION_TRAIN + filename
        copyfile(this_file, destination)
        
    for filename in filename_test:
        this_file = SOURCE + filename
        destination = DESTINATION_TEST + filename
        copyfile(this_file, destination)
        
def split_test(SOURCE, DESTINATION_TRAIN,  DESTINATION_TEST): # label
    for filename in filename_train:
        this_file = SOURCE + filename.replace(".jpg",".txt")
        destination = DESTINATION_TRAIN + filename.replace(".jpg",".txt")
        copyfile(this_file, destination)
        
    for filename in filename_test:
        this_file = SOURCE + filename.replace(".jpg",".txt")
        destination = DESTINATION_TEST + filename.replace(".jpg",".txt")
        copyfile(this_file, destination)
  
# #folder images        
source_images = './Images/'
destination_images_train = './train_data/images/train/'
destination_images_val = './train_data/images/val/'
split_train(source_images, destination_images_train, destination_images_val)

#folder labels
source_labels = './Labels/TXT format/'
destination_labels_train = './train_data/labels/train/'
destination_labels_val = './train_data/labels/val/'
split_test(source_labels, destination_labels_train, destination_labels_val)
