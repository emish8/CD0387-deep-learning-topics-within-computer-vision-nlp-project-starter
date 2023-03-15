#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
import time
import logging
import sys

#TODO: Import dependencies for Debugging andd Profiling
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import smdebug.pytorch as smd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling
from smdebug import modes
import smdebug.pytorch as smd

def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
          
    '''
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Average accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, validation_loader, criterion, optimizer,  epochs, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    c = 0
    for e in range(epochs):
        print('epoch',e)
        
        # training data
        hook.set_mode(smd.modes.TRAIN)
        model.train()
        for (data, target) in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            c += len(data)
        
        # validaiton data
        hook.set_mode(smd.modes.EVAL)
        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for (data, target) in validation_loader:
                outputs = model(data)
                loss = criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == target.data).item()
        total_accuracy = running_corrects / len(validation_loader.dataset)
        logger.info(f"Validation set: Average accuracy: {100*total_acc}%")
        
    return model    
    

# reference: Taken from course exercise
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    #train data loader
    train_path = os.path.join(data, "train")
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.ImageFolder(root=train_path, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    #test data loader
    test_path = os.path.join(data, "test")
    
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    testset = torchvision.datasets.ImageFolder(root=test_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    #validation data loader
    validation_path = os.path.join(data, "valid")
    validationset = torchvision.datasets.ImageFolder(root=validation_path, transform=test_transform)
    validation_loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=False)
   
    return train_loader, validation_loader, test_loader



def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    #device
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=net()
    model = model.to(dev)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
   
     
    # debugger hook
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    # calling loader 
    train_loader, validation_loader, test_loader = create_data_loaders(data=args.data, batch_size=args.batch_size)
    
    model=train(model, train_loader, validation_loader, loss_criterion, optimizer, args.epochs, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)
       

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    # ref: Course execeise Script Mode in SageMaker
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"],
        help="training data path in S3"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"],
        help="location to save the model to"
    )
    args=parser.parse_args()
    
    # logging information 

    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Test Batch Size: {args.test_batch_size}")
    logging.info(f"Epochs: {args.epochs}")
    
    main(args)
