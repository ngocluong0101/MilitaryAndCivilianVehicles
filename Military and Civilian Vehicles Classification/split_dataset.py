import os
from shutil import copyfile

import pandas as pd

df_label_train = pd.read_csv("./Labels/CSV Format/train_labels.csv")
df_label_test = pd.read_csv("./Labels/CSV Format/test_labels.csv")

filename_test = set(df_label_test["filename"].unique())
# Exclude any filename that also appears in the val/test set so train and val
# never overlap (the source CSVs ship with test_labels fully duplicated inside
# train_labels, which otherwise leaks val images into training).
filename_train = set(df_label_train["filename"].unique()) - filename_test

for d in (
    "./train_data",
    "./train_data/images",
    "./train_data/labels",
    "./train_data/images/train",
    "./train_data/images/val",
    "./train_data/labels/train",
    "./train_data/labels/val",
):
    os.makedirs(d, exist_ok=True)


def copy_files(filenames, source, destination):
    for filename in filenames:
        copyfile(source + filename, destination + filename)


# images
source_images = "./Images/"
copy_files(filename_train, source_images, "./train_data/images/train/")
copy_files(filename_test, source_images, "./train_data/images/val/")

# labels
source_labels = "./Labels/TXT format/"
copy_files(
    {f.replace(".jpg", ".txt") for f in filename_train},
    source_labels,
    "./train_data/labels/train/",
)
copy_files(
    {f.replace(".jpg", ".txt") for f in filename_test},
    source_labels,
    "./train_data/labels/val/",
)
