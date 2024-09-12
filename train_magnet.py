"""

As indicated in the model, the network used a pretrained intiailization
which is pretrained on ImageNet for 3 epochs only.

    We find that it is useful to warm-start any DML optimization with weights of a
    partly-trained a standard softmax classifier. It is important to not use weights
    of a net trained to completion, as this would result in information dissipation
    and as such defeat the purpose of pursuing DML in the first place. Hence, we
    initialize all models with the weights of a net trained on ImageNet
    (Russakovsky et al., 2015) for 3 epochs only. (page 8, section 4)

"""

import argparse
import os
import sys
import shutil
import time
import torch

import numpy                    as np
import torchvision.models       as models
import torchvision.transforms   as transforms
import torchvision.datasets     as datasets


import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from datasets                   import ImageNet, oxford_iiit_pet, oxford_flowers, magnet_MNIST, customCIFAR10
from models                     import magnetInception
from torchvision.models         import resnet18, ResNet18_Weights
from tensorboardX               import SummaryWriter    as Logger
from util.torch_utils           import to_var, save_checkpoint, AverageMeter, accuracy
from util                       import magnet_loss, triplet_loss, softkNN_metrics, softkNC_metrics
from torch.optim.lr_scheduler   import MultiStepLR
from IPython                    import embed
from sklearn.cluster            import KMeans
from torch.utils.data.sampler   import Sampler


import torch.nn as nn
import torch.nn.functional as F

def main(args):
    curr_time = time.time()

    print("#############  Read in Database   ##############")
    train_loader, valid_loader = get_loaders()

    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("######## Initiate Model and Optimizer   ##############")
    # Model - inception_v3 as specified in the paper
    # Note: This is slightly different to the model used by the paper,
    # however, the differences should be minor in terms of implementation and impact on results

    if args.dataset == "MNIST":
        model = Net(args.embedding_size)
    elif args.dataset == "CIFAR10":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.embedding_size)
        for name, param in model.named_parameters():
            if name.startswith("conv1.") or name.startswith("bn1.") or name.startswith("layer1.") or name.startswith("layer2.") or name.startswith("layer3.0."):
                param.requires_grad = False
    else:
        model = magnetInception(args.embedding_size)

    print("# total params:", sum(p.numel() for p in model.parameters()))
    print("# trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.resume is not None:
        print("Loading pretrained Module")
        checkpoint      = torch.load(args.resume)
        state_dict      = checkpoint['state_dict']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    model   = torch.nn.DataParallel(model).cuda()

    # Criterion was not specified by the paper, it was assumed to be cross entropy (as commonly used)
    if args.loss == "magnet":
        criterion = magnet_loss(D = args.D, M = args.M, alpha = args.GAP).cuda()    # Loss function
    elif args.loss == "triplet":
        criterion = triplet_loss(alpha = 0.304).cuda()    # Loss function
    elif args.loss == "softmax":
        criterion =     torch.nn.CrossEntropyLoss().cuda()    # Loss function
    else:
        print("Undefined Loss Function")
        exit()

    params  = list(model.parameters())                                      # Parameters to train

    # Optimizer -- the optimizer is not specified in the paper, and was ssumed to
    # be SGD. The parameters of the model were also not specified and were set
    # to commonly values used by pytorch (lr = 0.1, momentum = 0.3, decay = 1e-4)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    # The paper does not specify an annealing factor, we set it to 1.0 (no annealing)
    scheduler = MultiStepLR( optimizer,
                             milestones=list(range(0, args.num_epochs, 1)),
                             gamma=args.annealing_factor)

    print("Time taken:  {} seconds".format(time.time() - curr_time) )
    curr_time = time.time()

    print("#############  Start Training     ##############")
    total_step = len(train_loader)

    cluster_centers, cluster_assignment = indexing_step(    model = model,
                                                            data_loader = train_loader,
                                                            cluster_centers = None)

    loss_vector = 10. * np.ones(args.K * args.num_classes)
    loss_count  = np.ones(args.K * args.num_classes)

    for epoch in range(0, args.num_epochs):

        if args.evaluate_only:         exit()

        order = define_order(cluster_assignment, cluster_centers, loss_vector/loss_count)
        train_loader.dataset.update_read_order(order)

        logger.add_scalar("Misc/Epoch Number", epoch, epoch * total_step)
        loss_vector, loss_count, stdev = train_step(   model        = model,
                                    train_loader = train_loader,
                                    criterion    = criterion,
                                    epoch        = epoch,
                                    optimizer    = optimizer,
                                    step         = epoch * total_step,
                                    valid_loader = valid_loader,
                                    assignment   = cluster_assignment,
                                    loss_vector  = loss_vector,
                                    loss_count   = loss_count)

        logger.add_scalar(args.dataset + "/STDEV ",   stdev,   epoch * total_step)

        if epoch % args.eval_epoch == 0:
            curr_loss, curr_wacc = eval_step(   model       = model,
                                                data_loader = train_loader,
                                                criterion   = criterion,
                                                step        = epoch * total_step,
                                                datasplit   = "train",
                                                stdev       = stdev,
                                                cluster_centers = cluster_centers)

            curr_loss, curr_wacc = eval_step(   model       = model,
                                                data_loader = valid_loader,
                                                criterion   = criterion,
                                                step        = epoch * total_step,
                                                datasplit   = "valid",
                                                stdev       = stdev,
                                                cluster_centers = cluster_centers)

        if args.optimizer == 'sgd':    scheduler.step()

        cluster_centers, assignment = indexing_step( model = model, data_loader = train_loader, cluster_centers = cluster_centers)

        # args = save_checkpoint(  model      = model,
        #                          optimizer  = optimizer,
        #                          curr_epoch = epoch,
        #                          curr_loss  = curr_loss,
        #                          curr_step  = (total_step * epoch),
        #                          args       = args,
        #                          curr_acc   = curr_wacc,
        #                          filename   = ('model@epoch%d.pkl' %(epoch)))

        if epoch % args.save_epoch == 0:
            args = save_checkpoint(  model      = model,
                                    optimizer  = optimizer,
                                    curr_epoch = epoch,
                                    curr_loss  = curr_loss,
                                    curr_step  = (total_step * epoch),
                                    args       = args,
                                    curr_acc   = curr_wacc,
                                    filename   = ('model@epoch%d.pkl' %(epoch)))

def train_step(model, train_loader, criterion, optimizer, epoch, step, valid_loader = None, assignment = None, loss_vector = None, loss_count=None):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1    = AverageMeter()
    top5    = AverageMeter()
    stdev    = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, inst_indices) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)

        # compute output
        output = model(input_var)
        loss, c_loss_vector, c_loss_count, c_stdev = criterion(output, list(inst_indices.numpy()), assignment, loss_vector, loss_count, input, model)
        stdev.update(c_stdev, 1)

        loss_vector *= 0.9
        loss_count  *= 0.9
        loss_vector += (c_loss_vector/100.)
        loss_count  += c_loss_count

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        model.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        del loss, output
        batch_time.update(time.time() - end)

        if i % args.log_rate == 0 and i > 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            # Log items
            logger.add_scalar("Misc/batch time (s)",    batch_time.avg,                                   step + i)
            logger.add_scalar("Misc/Train_%",           1.0 - (data_time.avg/batch_time.avg),             step + i)
            logger.add_scalar("Misc/epoch time (min)",  batch_time.sum / 60.,                             step + i)
            logger.add_scalar("Misc/time left (min)",   ((len(train_loader) - i) * batch_time.avg) / 60., step + i)
            logger.add_scalar(args.dataset + "/Loss train  (avg)",          losses.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc5 train (avg)",          top5.avg,                      step + i)
            # logger.add_scalar(args.dataset + "/Perc1 train (avg)",          top1.avg,                      step + i)

        end = time.time()

        if valid_loader != None and i % args.eval_step == 0 and i > 0:
            _, _ = eval_step(   model       = model,
                                data_loader = valid_loader,
                                criterion   = criterion,
                                step        = step + i,
                                datasplit   = "valid")
            model.train()

    return loss_vector, loss_count, stdev.avg

def eval_step( model, data_loader,  criterion, step, datasplit, stdev, cluster_centers):
    batch_time = AverageMeter()
    losses = AverageMeter()

    metrics = softkNC_metrics(stdev = stdev, cluster_centers = cluster_centers ,K = args.K, L = args.L)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, inst_indices) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())

            # compute output
            output = model(input_var)

            # measure accuracy and record loss
            metrics.update(output, target)

    print("Evaluation: Forward Embedding Calculation (Time Elapsed {time:.3f})".format(time=time.time() - end))
    curr_time = time.time()
    acc1, acc5 = metrics.accuracy(stdev = stdev)

    print('Test: \t'
          'Time {test_time:.3f}\t'
          # 'Loss {loss.avg:.4f}\t'
          'Prec@1 {top1:.3f}\t'
          'Prec@5 {top5:.3f}'.format(
           i, len(data_loader), test_time=time.time() - curr_time, loss=losses
           , top1=acc1, top5=acc5)
    )

    # logger.add_scalar(args.dataset + "/Loss valid  (avg)",   losses.avg, step)
    logger.add_scalar(args.dataset + "/Perc5 " + datasplit + " (avg)",   acc5,   step)
    logger.add_scalar(args.dataset + "/Perc1 " + datasplit + " (avg)",   acc1,   step)

    return losses.avg, 0.0

def indexing_step(model, data_loader, cluster_centers):

        model.eval()
        curr_time = time.time()

        t_embeddings  = []
        t_labels      = []
        t_indices     = []

        data_loader.dataset.default_read_order()

        for i, (input, target, indices) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input, requires_grad=True)
            # compute output
            output = model(input_var)

            t_embeddings += list(output.cpu().data.numpy())
            t_labels     += list(target.numpy())
            t_indices    += list(indices.numpy())
            del input, target, indices

        # End FOR

        print('KNN: Forward Pass Time   {time:.3f}'.format(time=time.time() - curr_time))
        curr_time = time.time()
        cluster_centers, assignment = cluster_assignment(t_indices, t_embeddings, t_labels, init_centers= cluster_centers)
        print('KNN: Clustering time     {time:.3f}'.format(time=time.time() - curr_time))

        return cluster_centers, assignment

def cluster_assignment(indices, embeddings, labels, init_centers):
    cluster_nums    = args.num_classes * args.K

    assignment      = [-1] * len(indices)
    cluster_centers = np.zeros((cluster_nums, args.embedding_size))
    class_dict      = {}

    # Create a class dict and initialize it wiht all the embeddings
    for i in range(0, args.num_classes):
        class_dict[i] = {'embeddings' : [], 'indices' : []}
        for j in range(len(labels)):
            if labels[j] == i:
                class_dict[i]['embeddings'].append(embeddings[j])
                class_dict[i]['indices'].append(indices[j])

        # Convert list to a single array
        class_dict[i]['embeddings'] = np.asarray(class_dict[i]['embeddings'])

        #Calculate K-Means++ clustering
        if init_centers is None:
            c_init_centers = 'k-means++'
        else:
            c_init_centers = init_centers[i*args.K:(i+1)*args.K, :]

        kmeans = KMeans(n_clusters=args.K, init=c_init_centers, n_init="auto").fit(class_dict[i]['embeddings'])
        k_labels    = kmeans.labels_
        k_centers   = kmeans.cluster_centers_

        for k in range(0, args.K):
            global_k                   = (i * args.K) + k
            cluster_centers[global_k]  = k_centers[k]

            for l in range(0, class_dict[i]['embeddings'].shape[0] ):
                if k_labels[l] == k:
                    try:
                        assignment[ class_dict[i]['indices'][l] ] = global_k
                    except:
                        embed()

    if (-1 in assignment):
        embed()
    assert not (-1 in assignment), "Error: Not all indices were assigned!"

    return cluster_centers, assignment

def define_order(cluster_assignment, cluster_centers, loss_vector):

    num_clusters = args.num_classes * args.K

    cluster_distances = np.ones((num_clusters, num_clusters))

    for i in range(0, num_clusters):
        for j in range(i + 1, num_clusters):
            d = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            cluster_distances[i][j] = d
            cluster_distances[j][i] = d

    #construct useful dict
    cluster_dict = {}
    for i in range(0, num_clusters):
        cluster_dict[i] = []

    for i in range(0, len(cluster_assignment)):
        cluster_dict[cluster_assignment[i]].append(i)

    cluster_to_remove = []
    for i in range(0, num_clusters):
        if len(cluster_dict[i]) < 4:
            loss_vector[i] = 0.0
            cluster_to_remove.append(i)
            # print("Not sampling from cluster " + str(i) + " because it had less that 4 elements ")

    loss_vector = np.asarray(loss_vector)
    loss_vector = loss_vector / sum(loss_vector)
    loss_vector = list(loss_vector)

    NONZERO_CLUSTERS = num_clusters - len(cluster_to_remove)
    print("Number of proper clusters: " + str(NONZERO_CLUSTERS))

    # embed()
    order = []
    # construct batches
    for B in range(0, args.NUM_BATCHES):
        M0 = np.random.choice(range(0, num_clusters), p = loss_vector)
        # print("Cluster to deal with is " + str(M0))

        C0K0 = (M0 // args.K) * args.K      #first cluster index in that class

        bad_clusters = cluster_to_remove + list(range(C0K0, C0K0 + args.K))

        near_M = np.argsort(cluster_distances[M0])
        near_M = [m for m in near_M if m not in bad_clusters]    # Imposter clusters in order of closeness

        near_imposter = near_M[0:args.M-1]
        near_imposter.append(M0)

        if len(near_imposter) < args.M:
            # skip batch
            continue

        ignore_batch = False
        for Mi in near_imposter:
            if len(cluster_dict[Mi]) < args.D:
                ignore_batch = True

        if ignore_batch:
            continue

        for Mi in near_imposter:
            samplesMi = np.random.permutation(len(cluster_dict[Mi]))[0:args.D]
            my_s = []
            for s in samplesMi:
                order.append(cluster_dict[Mi][s])
                my_s.append(cluster_dict[Mi][s])

            for s in my_s:
                cluster_dict[Mi].remove(s)

            if len(cluster_dict[Mi]) < args.D:
                loss_vector[Mi] = 0.0
                loss_vector = loss_vector / np.sum(loss_vector)
                cluster_to_remove.append(Mi)
    return order

def get_loaders():

    # Data loading code (From PyTorch example https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if args.dataset == "CIFAR10":
        input_size = 224
    else:
        input_size = 299

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    print("Generating Datasets")
    if args.dataset == "oxford":
        train_dataset = oxford_iiit_pet("train", args.data_path, transform = train_transform)
        valid_dataset = oxford_iiit_pet("test",  args.data_path, transform = valid_transform)
    elif args.dataset == "flowers":
        train_dataset = oxford_flowers("train", args.data_path, transform = train_transform)
        valid_dataset = oxford_flowers("valid",  args.data_path, transform = valid_transform)
    elif args.dataset == "MNIST":
        train_dataset   =   magnet_MNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        valid_dataset = magnet_MNIST('../data', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    elif args.dataset == "CIFAR10":
        train_dataset = customCIFAR10(root='../data', train=True, download=True, transform=valid_transform)
        valid_dataset = customCIFAR10(root='../data', train=False, download=True, transform=valid_transform)
    else:
        print("No dataset generated!")
        exit()
    # END IF

    print("Generating Data Loaders")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        drop_last = False
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    return train_loader, valid_loader


class Net(torch.nn.Module):

    def __init__(self, embedding_size):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, embedding_size)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # logging parameters
    parser.add_argument('--save_epoch',         type=int , default=10)
    parser.add_argument('--eval_epoch',         type=int , default=3)
    parser.add_argument('--eval_step',          type=int , default=1000)
    parser.add_argument('--log_rate',           type=int, default=10)
    parser.add_argument('--workers',            type=int, default=7)
    parser.add_argument('--world_size',         type=int, default=1)

    # training parameters
    parser.add_argument('--num_epochs',         type=int,   default=100)
    parser.add_argument('--embedding_size',     type=int,   default=512)
    parser.add_argument('--lr',                 type=float, default=0.1)
    parser.add_argument('--momentum',           type=float, default=0.9)
    parser.add_argument('--weight_decay',       type=float, default=1e-4)
    parser.add_argument('--optimizer',          type=str,   default='sgd')
    parser.add_argument('--annealing_factor',   type=float, default=1.0)

    # experiment details
    parser.add_argument('--dataset',            type=str, default='oxford')
    parser.add_argument('--model',              type=str, default='inception')
    parser.add_argument('--experiment_name',    type=str, default= 'MagnetTest')
    parser.add_argument('--loss',               type=str, default= 'softmax')
    parser.add_argument('--evaluate_only',      action="store_true",default=False)
    parser.add_argument('--evaluate_train',     action="store_true",default=False)
    parser.add_argument('--resume',             type=str, default=None)

    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

    # Magnet Loss Parameters
    parser.add_argument('--M',                  type=int, default=8)       # Number of nearest clusters per mini-batch
    parser.add_argument('--K',                  type=int, default=8)       # Number of clusters per class
    parser.add_argument('--L',                  type=int, default=8)       # Number of clusters per class
    parser.add_argument('--D',                  type=int, default=4)       # Number of examples per cluster
    parser.add_argument('--GAP',                type=float, default=4)       # Number of examples per cluster
    parser.add_argument('--NUM_BATCHES',        type=int, default=100)       # Number of examples per cluster

    args = parser.parse_args()
    print(args)
    print("")

    args.batch_size = args.M * args.D

    if args.dataset == "oxford":
        args.data_path = "/z/home/mbanani/datasets/Oxford-IIIT_Pet/"
        args.num_classes = 37
    elif args.dataset == "MNIST":
        args.data_path = None
        args.num_classes = 10
    elif args.dataset == "flowers":
        args.data_path = "/z/home/mbanani/datasets/Oxford_Flowers/"
        args.num_classes = 102
    elif args.dataset == "CIFAR10":
        args.data_path = None
        args.num_classes = 10

    root_dir                    = os.path.dirname(os.path.abspath(__file__))
    experiment_result_dir       = os.path.join(root_dir, os.path.join('experiments',args.dataset))
    args.full_experiment_name   = ("exp_%s_%s_%s" % ( time.strftime("%m_%d_%H_%M_%S"), args.dataset, args.experiment_name) )
    args.experiment_path        = os.path.join(experiment_result_dir, args.full_experiment_name)
    args.best_loss              = sys.float_info.max
    args.best_acc               = 0.

    # Create model directory
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)

    print("Experiment path is : ", args.experiment_path)

    if args.M > (args.K * args.num_classes):
        print("Number of imposter clusters is larger than number of possible clusters. Setting it to NUM_CLUSTERS - 2")
        args.M = (args.K * args.num_classes) - 2

    # Define Logger
    tensorboard_logdir = '/z/home/mbanani/tensorboard_logs'
    log_name    = args.full_experiment_name
    logger      = Logger(log_dir = os.path.join(tensorboard_logdir, log_name))

    main(args)
