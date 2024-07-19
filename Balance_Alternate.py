import os
import math
import torch
import random
from itertools import chain
from DataLoader import MyDataLoader
from utility_functions import process_way

def Balance_Alternate(labels, nw, dic_total_, batch_size, cross_count_):
    train_images_path = []      # Store all image paths for the training set
    train_images_label = []     # Store the index information corresponding to the images of the training set   
    val_images_path = []        # Store all picture paths for the validation set  
    val_images_label = []       # Store the index information corresponding to the image of the verification set
    dic_image_label = {}
    # Obtain data by fold and traverse the dictionary
    for key_,value_ in dic_total_.items():
        value_temp = value_.copy()
        category_data_  = len(value_)
        category_step = (category_data_ / 4)
        begin_ = math.floor((cross_count_ - 1) * category_step)
        if cross_count_ == 4:
            end_ = category_data_
        else:
            end_ = math.floor(begin_ + category_step)

        for path_ in value_[begin_:end_]:
            val_images_path.append(path_)
            val_images_label.append(key_)
            value_temp.remove(path_)
        image_ = []             # Store every category of workouts
        for path_ in value_temp:
            image_.append(path_)
        dic_image_label[key_] = image_

    label_img = {}
    label_img_count = {}
    max_img_count = 0
    category_batch = batch_size // len(labels)      
    remainder = batch_size - (category_batch*len(labels))

    for keys_,values_ in dic_image_label.items():
        image_folder = values_  # All images in the current category folder                 
        if len(image_folder) > max_img_count:
            max_img_count = len(image_folder)
        image_list = []
        for image_dir in image_folder:          # Traverse the images          
            image_list.append(os.path.join(image_folder, image_dir))
        label_img[keys_] = image_list
        label_img_count[keys_] = category_batch

    dic_count = {}
    for key_,value_ in label_img.items():
        dic_count[key_] = len(value_)
    dic_count = dict(sorted(dic_count.items(), key=lambda x: x[1], reverse=True))

    if remainder != 0:    
        for key_,value_ in dic_count.items():   # Determine the number of samples for each class in a batch 
            label_img_count[key_] += 1
            remainder -=1 
            if remainder == 0:
                break

    batch = math.ceil(max_img_count / max(list(label_img_count.values())))
    for key_,value_ in label_img.items():
        while len(value_) < label_img_count[key_] * batch:
            random_element = random.choice(value_)
            value_.append(random_element)
        label_img[key_] = value_
    # After random filling, batch is made
    for batch_ in range(batch):
        img_path = []
        lab_img = []
        for key_,value_ in label_img.items():
            begin_idx = batch_ * label_img_count[key_]
            end_idx = begin_idx + label_img_count[key_]
            for idx in range(begin_idx,end_idx):
                img_path.append(value_[idx])
                lab_img.append(key_)
        # The list is randomly scrambled
        zip_ = list(zip(img_path,lab_img))
        random.shuffle(zip_)
        img_path,lab_img = zip(*zip_)

        train_images_path.append(img_path)
        train_images_label.append(lab_img)

    train_images_path = list(chain(*train_images_path))
    train_images_label = list(chain(*train_images_label))

    train_info = [train_images_path, train_images_label]
    val_info = [val_images_path, val_images_label]

    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info
    # Instantiate the training dataset
    train_data_set = MyDataLoader(images_path=train_images_path,
                                images_class=train_images_label,
                                )
    # Instantiate the validation dataset
    val_data_set = MyDataLoader(images_path=val_images_path,
                                images_class=val_images_label,
                                )
    # Each process corresponding to rank is assigned the trained sample index
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    # Assemble a list of each batch_size element of the sample index
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=False)

    train_loader = torch.utils.data.DataLoader(train_data_set,
                                                batch_sampler=train_batch_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn = process_way
                                                )
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                                batch_size=batch_size,
                                                sampler=val_sampler,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn = process_way
                                            ) 
    
    return train_sampler, train_loader, val_sampler, val_loader