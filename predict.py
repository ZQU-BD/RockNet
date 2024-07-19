import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from RockNet import RockNet
from prettytable import PrettyTable
from torchvision import transforms, datasets

def collate_fn(batch):
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
        processed_image0 = data_transform(image)
        resized_images.append(processed_image0)
        # Local information
        flag_num = 1
        for i in range(2):
            for j in range(2):
                eval_name_ = eval('resized_images' + str(flag_num))
                processed_image1 = image.crop((int(j*1*width/4),int(i*1*height/4),int((j*1+3)*width/4),int((i*1+3)*height/4)))
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

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # Calculate the accuracy
        sum_TP = np.trace(self.matrix)
        total_samples = np.sum(self.matrix)
        acc = sum_TP / total_samples
        print("The model accuracy is ", acc)

        # Initialize variables for total precision, recall, specificity, and F1-score
        total_precision = 0
        total_recall = 0
        total_specificity = 0
        total_f1 = 0

        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1-Score"]

        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = total_samples - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.0
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.0
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.0
            F1 = round(2 * (Precision * Recall) / (Precision + Recall), 3) if (Precision + Recall) != 0 else 0.0

            # Accumulate values for total calculation
            total_precision += Precision
            total_recall += Recall
            total_specificity += Specificity
            total_f1 += F1

            # Add row to table
            table.add_row([f"Class {i}", Precision, Recall, Specificity, F1])

        # Calculate average values
        avg_precision = round(total_precision / self.num_classes, 3)
        avg_recall = round(total_recall / self.num_classes, 3)
        avg_specificity = round(total_specificity / self.num_classes, 3)
        avg_f1 = round(total_f1 / self.num_classes, 3)

        # Add row for average values
        table.add_row(["Average", avg_precision, avg_recall, avg_specificity, avg_f1])
        # Print the table
        print(table)

    def plot(self):
        matrix = self.matrix
        print('matrix is :')
        # Print the confusion matrix
        print(matrix)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('The testing device is: {}'.format(device))

    batch_size = 12
    num_classes = 16                        # The category of the dataset at the time of training
    model_weight_path = "./model_best.pth"
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes = num_classes, labels = labels)

    # The catalog of the test dataset
    image_path = "./images"
    validate_dataset = datasets.ImageFolder(root=image_path+r"/test")
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, 
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  collate_fn = collate_fn
                                                  )

    model = RockNet(num_classes=16).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device),strict=False)
    model.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images,val_images1,val_images2,val_images3,val_images4, val_labels = val_data
            pred = model(val_images.to(device))  
            pred1 = model(val_images1.to(device))
            pred2 = model(val_images2.to(device))
            pred3 = model(val_images3.to(device))
            pred4 = model(val_images4.to(device))
            
            pred = pred*0.4 + pred1*0.15 + pred2*0.15 + pred3*0.15 + pred4*0.15
            
            outputs = torch.softmax(pred, dim=1)
            outputs = torch.argmax(pred, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()