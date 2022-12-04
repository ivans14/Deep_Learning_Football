import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from Dataset import img_dataset, get_transform, create_datasets, create_datasetClases, show_random_image_boxes

import files.utils as utils
import files.transforms as T
from files.engine import train_one_epoch, evaluate


def get_object_detection_model(num_classes):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def train():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.empty_cache() # To clear the cache in Cuda so we have more room for the data to upload into GPU.

    train_labels, train_images, test_labels, test_images = create_datasets()
    train_ds, val_ds, _ = create_datasetClases(train_labels,train_images,test_labels,test_images)

    # split the dataset in train and validation set
    torch.manual_seed(1)
    indices = torch.randperm(len(train_ds)).tolist()
    split_size = int(len(train_ds)*0.2)
    ds = torch.utils.data.Subset(train_ds, indices[:-split_size])
    val_ds = torch.utils.data.Subset(val_ds, indices[-split_size:])

    data_loader_train = torch.utils.data.DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    data_loader_val = torch.utils.data.DataLoader(
        val_ds,
        batch_size=0,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    num_classes = 3 #CLASS ZERO FOR BACKGROUND, 1 FOR PLAYERS AND 2 FOR BALL 

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # Define and construct optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler for the optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    num_epochs = 4

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)


    # Save model (i.e. its weights)
    models_dir = "trained_models/"
    torch.save(model.state_dict(), models_dir + 'CNN_weights_{}.pth'.format(len([entry for entry in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir,entry))])))
    print("Finished training.")

if __name__ == '__main__':
    train()
    print("Training successful :)")