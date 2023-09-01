import os
from pathlib import Path
import shutil

# Remove images with no objects in them
def remove_empty_images(files_list, labels_path, label_ext):
    updated_file_list = []
    for file in files_list:
        with open(os.path.join(labels_path, file+label_ext), 'r') as f:
            data = f.read().rstrip()
            if len(data) > 0:
                updated_file_list.append(file)
    return updated_file_list

# Separate the samples in 3 sets creating 3 directories: train, validation, test
# In each directory copy a random subset of the total data available, without repetitions and with fixed proportions.
# At the end we will have the 3 directories each one containing two directories images and labels
def split_dataset(images_path,
                  img_ext,
                  labels_path,
                  label_ext,
                  new_dataset_path,
                  ratios=[0.75, 0.15, 0.10],
                  names=["train", "validation", "test"]):
    # Obtain the list of files inside the dataset (Path.stem remove the extension from the file name)
    files_list = [Path(img_name).stem for img_name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, img_name))]
    
    # Do not consider images with no objects in them
    #files_list = [file for file in files_list if os.stat(os.path.join(labels_path, file+label_ext)).st_size != 0]
    files_list = remove_empty_images(files_list, labels_path, label_ext)
    
    #DEBUG
    #files_list = files_list[0:1000] #DEBUG
    
    # Split the files (eventually, you can also first shuffle the files)
    splits_list = []
    last_val_prec = 0
    for split_ratio in ratios:
        last_val = last_val_prec + int(len(files_list) * split_ratio)
        split = files_list[last_val_prec:last_val]
        splits_list.append(split)
        last_val_prec = last_val

    # Create the directories and copy both images and labels inside
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
        for i, split in enumerate(splits_list):
            split_path = os.path.join(new_dataset_path, names[i])
            os.mkdir(split_path)
            new_images_path = os.path.join(split_path, "images")
            new_labels_path = os.path.join(split_path, "labels")
            os.mkdir(new_images_path)
            print(f"created: {new_images_path}")
            os.mkdir(new_labels_path)
            print(f"created: {new_labels_path}")
            for file_name in split:
                old_img_path = os.path.join(images_path, file_name+img_ext)
                new_img_path = os.path.join(new_images_path, file_name+img_ext)
                shutil.copyfile(old_img_path, new_img_path)
                old_label_path = os.path.join(labels_path, file_name+label_ext)
                new_label_path = os.path.join(new_labels_path, file_name+label_ext)
                shutil.copyfile(old_label_path, new_label_path)
    else:
        print("Dataset already exists. Delete it before proceding.")

