import torch
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

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
                labels.append(self.labels_list[line][7])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        print(boxes)
        print(boxes.shape)
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
                A.SmallestMaxSize(shift_limit = 0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
                ToTensorV2(p=1.0)
            ],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )
    else:
        return A.Compose(
            [ToTensorV2(p=1.0)],
            bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}
        )




