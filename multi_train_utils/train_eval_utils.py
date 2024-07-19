import sys
import torch
from tqdm import tqdm
from multi_train_utils.distributed_utils import reduce_value, is_main_process

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    # Print the training progress in process 0
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images,images1,images2,images3,images4,labels = data

        pred = model(images.to(device))
        loss = loss_function(pred, labels.to(device))
        pred1 = model(images1.to(device))
        loss1 = loss_function(pred1, labels.to(device))
        pred2 = model(images2.to(device))
        loss2 = loss_function(pred2, labels.to(device))
        pred3 = model(images3.to(device))
        loss3 = loss_function(pred3, labels.to(device))
        pred4 = model(images4.to(device))
        loss4 = loss_function(pred4, labels.to(device))
    
        loss = loss*0.4 + loss1*0.15 + loss2*0.15 + loss3*0.15 + loss4*0.15 
        loss.backward()
        
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        # Print the mean loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    # Used to store the number of samples that are correctly predicted
    sum_num = torch.zeros(1).to(device)

    # Print the validation progress in process 0
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images,images1,images2,images3,images4,labels = data
        pred = model(images.to(device))
        pred1 = model(images1.to(device))
        pred2 = model(images2.to(device))
        pred3 = model(images3.to(device))
        pred4 = model(images4.to(device))

        pred = pred*0.4 + pred1*0.15 + pred2*0.15 + pred3*0.15 + pred4*0.15
        
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()
        
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()
