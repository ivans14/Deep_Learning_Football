import torch
import numpy as np
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import random
import matplotlib.pyplot as plt

# Create a Dataset class
class img_dataset(torch.utils.data.Dataset):
    def __init__(self, height, width, dir_img, labels_list, images, transforms=None, mode='train') -> None:
        self.transforms = transforms
        self.height = height
        self.width = width
        self.dir_img = dir_img
        self.labels_list = labels_list
        self.images=images

    def __getitem__(self,id):
        boxes = []
        labels = []
        img_name = self.images[id]
        

        # reading the images and converting them to correct size and color    
        img = cv2.imread(img_name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        
        for line in range(1, len(self.labels_list)):
            #parsed = [float(x) for x in line.split(',')]
            if int(self.labels_list[line][0]) == int(self.images[id][-10:-4]):
                x = self.labels_list[line][2]
                y = self.labels_list[line][3]
                width = self.labels_list[line][4]
                height = self.labels_list[line][5]
                x_max = x + width
                y_max = y + height
                boxes.append([x,y,x_max,y_max])
                if int(self.labels_list[line][-1]) == -1:
                    labels.append(1)
                elif int(self.labels_list[line][-1]) == 1:
                    labels.append(2)
                else:
                    labels.append(self.labels_list[line][-1])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #print(boxes)
        #print(boxes.shape)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target["iscrowd"] = iscrowd
        target["area"] = area
        image_id = torch.tensor([id])
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transforms:
            sample = self.transforms(image = img_res,
                                        bboxes = target['boxes'],
                                        labels = labels)
            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            return img_res, target

        return img_res,boxes
         

    def __len__(self):
        return len(self.images)

# Create transformations to increase the dataset
def get_transform(train):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(0.5),
                #A.SmallestMaxSize(shift_limit = 0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
                #A.RandomBrightnessContrast(p=0.5),
                #A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
                ToTensorV2(p=1.0)
            ],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )
    else:
        return A.Compose(
            [ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )

def create_datasets():
    directory_train = 'proj_test/Train'

    # Load Train data
    train_labels = pd.read_csv(directory_train + "/proj_det/det.txt", sep=',')
    train_images = directory_train + "/proj_img1/{}.jpg"
    train_images = [train_images.format(str(i).zfill(6)) for i in range(1,len(train_labels['frame'].unique()))] # train_labels.index.values
    train_labels = [train_labels.columns.values.tolist()] + train_labels.values.tolist() # Convert DF to list

    #print(train_labels[0:5][0:7])

    # Load Test data
    directory_test = 'proj_test/Test'

    test_labels = pd.read_csv(directory_test + '/proj_det/det.txt', sep=',', index_col=0)
    test_images = directory_test + "/proj_img1/{}.jpg"
    test_images = [test_images.format(str(i).zfill(6)) for i in range(1,751)]
    test_labels = [test_labels.columns.values.tolist()] + test_labels.values.tolist()

    return train_labels, train_images, test_labels, test_images

def create_datasetClases(train_labels,train_images,test_labels,test_images):
    directory_train = 'proj_test/Train'
    directory_test = 'proj_test/Test'

    train_ds = img_dataset(1080,1920,directory_train + "/proj_img1/",train_labels,train_images, transforms=get_transform(train=True))
    val_ds = img_dataset(1080,1920,directory_train + "/proj_img1/",train_labels,train_images, transforms=get_transform(train=False))
    test_ds = img_dataset(1080,1920,directory_test + "/proj_img1/",test_labels,test_images, mode='test', transforms=get_transform(train=False))
    
    return train_ds, val_ds, test_ds

# To show a random train image and its bboxes
def show_random_image_boxes(img_dataset):
    idx_2print = random.randint(1, len(img_dataset.images))
    ran_img, boxes = img_dataset[idx_2print]

    img_2plot = ((ran_img.permute(1,2,0)).numpy()).copy()

    fig = plt.figure(figsize=(10,8))
    boxes_list = (boxes.get("boxes")).tolist()
    target_list = (boxes.get("labels")).tolist()   

    for i in range((boxes.get("boxes").shape)[0]):
        x = int(boxes_list[i][0])
        y = int(boxes_list[i][1])
        x_max = int(boxes_list[i][2])
        y_max = int(boxes_list[i][3])
        
        if (target_list[i] == 2): # Rectangle for ball in blue
            cv2.rectangle(img_2plot, (x,y),(x_max,y_max),(0,0,255),6)
            cv2.putText(img= img_2plot, text = str(target_list[i]), org = (x, y),fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 2, color = (0,0,255), thickness= 2, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(img_2plot, (x,y),(x_max,y_max),(255,0,0),5)
            cv2.putText(img= img_2plot, text = str(target_list[i]), org = (x, y),fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 2, color = (255,0,0), thickness= 2, lineType=cv2.LINE_AA)

    plt.imshow(img_2plot)
    plt.axis('off')
    plt.title("Figure num. " + str(((boxes.get("image_id")).tolist())[0]))
    plt.tight_layout(pad=1)
    plt.show()

if __name__ == '__main__':

    train_labels, train_images, test_labels, test_images = create_datasets()
    train_ds, test_ds = create_datasetClases(train_labels,train_images,test_labels,test_images)

    show_random_image_boxes(train_ds)
    print("Data imported successfully :)")