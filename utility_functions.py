import torch
import random
import numpy as np
from torchvision import transforms
auto_augment = transforms.AutoAugment()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def process_way(batch):
    # Data preprocessing method
    data_transform = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # Get the batch data
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Initialize variables and store global local information
    resized_images = []
    resized_images1 = []
    resized_images2 = []
    resized_images3 = []
    resized_images4 = []

    for image in images:
        # Get the size of image
        height = image.size[0]
        width = image.size[1]
        # Global information
        processed_image0 = auto_augment(image)
        processed_image0 = data_transform(processed_image0)
        resized_images.append(processed_image0)
        # Local information
        flag_num = 1
        for i in range(2):
            for j in range(2):
                eval_name_ = eval('resized_images' + str(flag_num))
                processed_image1 = image.crop((int(j*1*width/4),int(i*1*height/4),int((j*1+3)*width/4),int((i*1+3)*height/4)))
                processed_image1 = auto_augment(processed_image1)
                processed_image1 = data_transform(processed_image1)
                eval_name_.append(processed_image1)
                flag_num += 1
    resized_images = torch.stack(resized_images)
    resized_images1 = torch.stack(resized_images1)
    resized_images2 = torch.stack(resized_images2)
    resized_images3 = torch.stack(resized_images3)
    resized_images4 = torch.stack(resized_images4)
    labels = torch.tensor(labels)
    # Returns global and local information, as well as their corresponding labels
    return resized_images,resized_images1,resized_images2,resized_images3,resized_images4,labels