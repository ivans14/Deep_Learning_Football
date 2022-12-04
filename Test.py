import numpy as np
import pandas as pd
import torch
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Train import get_object_detection_model, models_dir
from Dataset import img_dataset, get_transform, Data
import files.utils as utils

def main():
    # Load latest trained model
    model = get_object_detection_model(num_classes=2)
    model.load_state_dict(torch.load(models_dir + "CNN_weights_{}.pth".format(len([entry for entry in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir,entry))]))))
    model.eval()

    #import test dataset
    directory_test = 'proj_test/test'
    test_labels = pd.read_csv(directory_test + '/proj_det/det.txt', sep=',', index_col=0)
    test_images = directory_test + "/proj_img1/{}.jpg"
    test_images = [test_images.format(str(i).zfill(6)) for i in test_labels.frame.values]
    test_labels = [test_labels.columns.values.tolist()] + test_labels.values.tolist()
    test_ds = img_dataset(1080,1920,directory_test + "/proj_img1/",test_labels,test_images, mode='test', transforms=get_transform(train=False))
    test_ds = img_dataset(1080,1920,directory_test + "/proj_img1/",test_labels,test_images, mode='test', transforms=get_transform(train=False))

    data_loader_test = torch.utils.data.DataLoader(
        test_ds,
        batch_size = 1,
        shuffle=False,
        num_workers=4,
        collate_fn =utils.collate_fn
    )


    samples = len(test_ds) + 7
    rows = samples // 8
    figure, ax = plt.subplots(nrows=rows, ncols = 8, figsize=(24,16))
     
    for i, (img, labels) in enumerate(data_loader_test):
        output = model(img)
        index = output.data.cpu().numpy().argmax()
        pil_img = return_transform(img.squeeze(0))
        cv_img = np.array(pil_img)

        ax.ravel()[i].imshow(cv_img)
        ax.ravel()[i].set_axis_off()
        


if __name__ == '__main__':
    main()