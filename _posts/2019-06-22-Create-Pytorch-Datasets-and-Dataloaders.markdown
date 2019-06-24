---
layout: post
title: "How to create and use custom PyTorch Dataset with ImageNet dataset example"
---

![thumb](/img/dataloader/thumb1.png)

In this post, I will tell about Pytorch Datasets and DataLoaders. As an example, I will create Dataset from a folder of images. This solution would work for ImageNet as well as OpenImage dataset as long as it would have the required folder structure.

[Here is gist link](https://gist.github.com/mf1024/a9199a325f9f7e98309b19eb820160a7) to the Dataset implementation which I will use throughout this post.

## Raw data preparation

Raw data preparation is the first step before deep learning models can be effectively trained and used.
Raw data comes in many different shapes and forms but deep learning models expect very specific input.

### Some raw data preparation tasks are:
- **Indexing of the samples.** Indexing is assigning each sample from the dataset a number from 0 to N-1 and providing a method to access each of the samples by its index. In it's simplest form indexing could be reading the data in memory array and accessing it during training and testing, but with larger datasets you won't be able to read everything into memory but you can provide a methods for accessing each sample by its index quickly - it can be just knowing the sample location in the filesystem.
- **Preparation of the shape of the data.** Deep learning models expect a very specific shape of the data as its input. For example, image dataset can contain images with many different resolutions. Before feeding them into the model, you will have to rescale and crop the images to the same sizes.
- **Packing the samples into mini-batches**
- **Data augmentation**
- **Data normalization**
- **Separation of the dataset into train, test and validation splits**


## The Dataset and DataLoader classes

PyTorch comes with *utils.data* which includes *Dataset* and *DataLoader* classes that handle raw data preparation tasks. Those classes allow you to abstract from details of data preparation when training and testing deep learning models.

**Dataset** is the parent class for creating custom Datasets. For CustomDataset to function together with DataLoader you only have to override two methods:
- **__getitem__(sample_index)** - must return on sample with index *sample_index*
- **__len__()** - must return the number of samples in the Dataset

**DataLoader** takes any Dataset instance as an argument and packs all of the data into batches so that each sample appears exactly once over all of the batches. Dataloader also creates an iterator so that it's easy to loop over the data.

### Here is a brief example of Dataset and DataLoader usage:

{% highlight python %}

from torch.utils.data import DataLoader, Dataset
BATCH_SIZE = 32

class CustomDataset(Dataset):
    def __init__(self):
        #Index the data
    def __getiten__(self, sample_index):
        return data[sample_index]
    def __len__(self):
        return len(data)

custom_dataset = CustomDataset()
custom_data_loader = DataLoader(custom_dataset, BATCH_SIZE)

#Train/Test
for batch in custom_data_loader:
    x = batch['sample']
    y = batch['labels']

...

{% endhighlight %}

# ImageNetDataset example

I have a bunch of images downloaded with [Imagenet Downloader](https://github.com/mf1024/ImageNet-Datasets-Downloader) in the following folder structure:

{% highlight shell %}
imagenet_images/
|-- apple/
|   |-- apple_1.jpg
|   |-- apple_2.jpg
|   |-- apple_3.jpg
|   ...
|-- car/
|   |-- car_1.jpg
|   |-- car_2.jpg
|   |-- car_3.jpg
|   ...
|-- dog/
|   |-- dog_1.jpg
|   |-- dog_2.jpg
|   ...
...

{% endhighlight %}

And I want to prepare this data for training and testing in PyTorch. You can take a look at the [code](https://gist.github.com/mf1024/a9199a325f9f7e98309b19eb820160a7) as I go through some of the fragments to explain them.

First, in the **ImageNetDataset.__init__()**, I collect all of the classes from the image data folder structure:
{% highlight python %}
for class_name in os.listdir(data_path):
    if not os.path.isdir(os.path.join(data_path,class_name)):
        continue
    self.classes.append(
       dict(
           class_idx = class_idx,
           class_name = class_name,
       ))
    class_idx += 1

{% endhighlight %}

To index the data, I walk through each of the class folder and store path of each image in the image_list. 

{% highlight python %}

self.image_list = []
for cls in self.classes:
    class_path = os.path.join(data_path, cls['class_name'])
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        self.image_list.append(dict(
            cls = cls,
            image_path = image_path,
            image_name = image_name,
        ))

{% endhighlight %}



I don't have two separate sets of images for training and testing, so I will have to do the splitting with ImageNetDataset class. To make the splits I will create two different ImageNetDataset instances where train and test datasets use two non-overlapping sets of images from the data.

To achieve this I introduce **img_idxes** list in the ImageNetDataset class, which I create with numbers from 0 to N-1, where N is the number of all samples in the data. Then I will shuffle the **img_idxes** randomly (have to make sure that the shuffle is done with the same random seed in the test instance and in the train instance so that the sets do not overlap) and I will keep the first 90% of the **img_idxes** in the case of train instance and last 10% of the **img_idxes** in the case of test instance. And then when accessing the image info from the image_list - I will use indexes from **img_idxes**.

{% highlight python %}

self.img_idxes = np.arange(0,len(self.image_list))

np.random.seed(random_seed)
np.random.shuffle(self.img_idxes)

last_train_sample = int(len(self.img_idxes) * train_split)
if is_train:
    self.img_idxes = self.img_idxes[:last_train_sample]
else:
    self.img_idxes = self.img_idxes[last_train_sample:]

{% endhighlight %}

**ImageNet.__init__()** argument **is_train** indicates if this instance is train or test instance,
**train_split** indicates the data amount split ratio of the train instance and test instance,
and **random_seed** is that random seed which is used for the random shuffle of the indexes.

{% highlight python %}
class ImageNetDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        is_train, 
        train_split = 0.9, 
        random_seed = 42, 
        target_transform = None, 
        num_classes = None
    ):
{% endhighlight %}

To create both of the instances, I encapsulate their creation in a single function:

{% highlight python %}

def get_imagenet_datasets(num_classes = None):

    random_seed = int(time.time())

    dataset_train = ImageNetDataset(
        IMAGENET_PATH, 
        is_train = True, 
        train_split = 0.9,
        random_seed=random_seed, 
        num_classes = num_classes
    )

    dataset_test = ImageNetDataset(
        IMAGENET_PATH, 
        is_train = False, 
        train_split = 0.9,
        random_seed=random_seed, 
        num_classes = num_classes
    )

    return dataset_train, dataset_test

{% endhighlight %}

In the **__len__()** of the ImageNetDataset I simply return the size of **img_idxes**.

{% highlight python %}
def __len__(self):
    return len(self.img_idxes)
{% endhighlight %}


In the **__getitem__(index)** function I get the index from the **img_idxes** list and then get the corresponding element from the **image_list**. Then I get the path of the image and read the image from the filesystem. 

{% highlight python %}

def __getitem__(self, index):

    img_idx = self.img_idxes[index]
    img_info = self.image_list[img_idx]
    img = Image.open(img_info['image_path'])

{% endhighlight %}

[torchvision](https://pytorch.org/docs/stable/torchvision/index.html) includes many [image transformation functions](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-pil-image) which you can use for resizing, cropping, flipping, doing random rotations and many others that you can use for image preparation and data augmentation.

The dataset I use contains images in many different resolutions. I have to do some transformations to make them all the same size. I first do resize. I scale it up if it is too small and I resize it smaller if it is too big. Then I do random cropping to the exact image size I have set for the dataset. I also make sure that the image has exactly 3 channels.

And then I return *dict()* of:
1. The cropped image
2. The class number
3. The class name

{% highlight python %}

return dict(
    image = img, 
    cls = img_info['cls']['class_idx'], 
    class_name = img_info['cls']['class_name']
)

{% endhighlight %}

The *__getitem__()* function will be used by the Dataloader to gather together batches of samples. 
Dataloader is smart enough to stack each of the dict() element together into batches. For example, if the *ImageNetDataset.__getitem__(index)* returns an dictionary with *image* element of shape *(3,64,64)* then the *batch['image']* from *DataLoader* which uses *ImageNetDataset* will contain element of shape *(BATCH_SIZE,3,64,64)*. And if *cls* element is an integer then *batch['cls']* will contain *BATCH_SIZE* elements with the corresponding classes of the batch images.


## Testing and plotting

I wrote a small script to show the use of DataLoader and plot some of the images from the first batch:

{% highlight python %}
dataset_train, dataset_test = get_imagenet_datasets(data_path)

print(f"Number of train samples: {dataset_train.__len__()}")
print(f"Number of test samples: {dataset_test.__len__()}")

BATCH_SIZE = 12

data_loader_train = DataLoader(
    dataset_train, 
    BATCH_SIZE, 
    shuffle = True
)

data_loader_test = DataLoader(
    dataset_test, 
    BATCH_SIZE, 
    shuffle = True
)


import matplotlib.pyplot as plt

fig, axes = plt.subplots(BATCH_SIZE//3,3, figsize=(6,10))

for batch in data_loader_train:

    print(f"Shape of batch['image'] {batch['image'].shape}")
    print(f"Shape of batch['cls'] {batch['cls'].shape}")

    for i in range(BATCH_SIZE):

        col = i % 3
        row = i // 3

        img = batch['image'][i].numpy()

        axes[row,col].set_axis_off()
        axes[row,col].set_title(batch['class_name'][i])
        axes[row,col].imshow(np.transpose(img,(1,2,0)))

    plt.show()
    break

{% endhighlight %}

I get the output:

{% highlight shell %}
Number of samples in train split 229485
Number of samples in test split 25499
Shape of batch['image'] torch.Size([12, 3, 128, 128])
Shape of batch['cls'] torch.Size([12])
{% endhighlight %}

Seems good. The ratio is 0.9 to 0.1, and the shapes are also right. Now let's see the actual images from the first batch:

And I get the following image plot:

![test_output](/img/dataloader/test_output.png)

