import os
import math
import json
import torch
import tempfile
import argparse
import torch.optim as optim
from RockNet import RockNet
from utility_functions import set_seed
import torch.optim.lr_scheduler as lr_scheduler
from Balance_Alternate import Balance_Alternate
from torch.utils.tensorboard import SummaryWriter
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup

def main(args):
    set_seed(100)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("Not find any GPU devices for training.")
    
    # Initialize the process environment
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    weights_path = args.weights_sample
    args.lr *= args.world_size  # The learning rate is multiplied based on the number of parallel GPUs
    result_folder = os.path.join('.', args.experiment_name)
    checkpoint_path = ""
    json_label_path = args.json_path
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)
    dic = {value: key for key, value in class_indict.items()}
    labels = [label for _, label in class_indict.items()]

    if rank == 0:               # The information is printed in the first process and the tensorboard is instantiated
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # Number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    
    # Instantiate the model
    model = RockNet(num_classes=args.num_classes).to(device)
    print(model)

    if os.path.exists(weights_path):        # If pre-training weight is present
        weights_dict = torch.load(weights_path, map_location=device)
        load_weights_dict = {k: v for k, v in weights_dict.items()
                             if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)
    else:                                   # The pre-training weight is not exist
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # Save the weights in the first process and then load them in the other processes to keep the initialization weights consistent
        if rank == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Whether to freeze weights
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # All but the last fully connected layer are frozen
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        if args.syncBN:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # Switch to DDP model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],broadcast_buffers=False)

    # Define the optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)

    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.
    dic_total_ = {}                 # The dictionary holds the data path for each category
    root = args.data_path
    train_path_folder = root + r"/train_val_sum"    # A collection of training and validation sets
    
    train_folder = os.listdir(train_path_folder)
    train_folder.sort()

    cross_count_ = 1                # Folds
    for folder in train_folder:
        path = os.path.join(train_path_folder,folder)
        image_folder = os.listdir(path)
        ls = []
        for image_dir in image_folder:
            ls.append(os.path.join(path,image_dir) )
        dic_total_[int(dic[folder])] = ls           # Dictionary holds data "Category : Path"
    
    cross_count_ = 1
    train_sampler, train_loader, val_sampler, val_loader = Balance_Alternate(labels, nw, dic_total_, batch_size, cross_count_)

    steps = int(args.epochs / 4)
        
    # Start training
    for epoch in range(args.epochs):
        if epoch != 0:
            # Determine whether need to alternate between training sets and validation sets
            if epoch % steps == 0:
                train_sampler, train_loader, val_sampler, val_loader = Balance_Alternate(labels, nw, dic_total_, batch_size, cross_count_)
                cross_count_ += 1
    
        train_sampler.set_epoch(epoch)
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()
        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size
        if rank == 0:
            print("[val epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.module.state_dict(), os.path.join(result_folder, 'model_best.pth'))
                print('\t\t【save model】')
            if epoch == (args.epochs-1):
                torch.save(model.module.state_dict(), os.path.join(result_folder, 'last.pth'))

    # Delete the temporary cache file
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
            
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    parser.add_argument('--json_path', type=str, default="./the_json_about_categories")
    parser.add_argument('--experiment_name', type=str, default="RockNet")
    # Whether or not to enable SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)        
    # The root directory where the dataset resides
    parser.add_argument('--data_path', type=str, default="/opt/data/private/Rock_1/train_multi_GPU/train_val_test")
    parser.add_argument('--weights_sample', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    '''
    The number of enabled processes (note that it is not a thread) 
    does not need to be set this parameter, and will be automatically set 
    according to the nproc_per_node
    '''    
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()
    main(opt)
