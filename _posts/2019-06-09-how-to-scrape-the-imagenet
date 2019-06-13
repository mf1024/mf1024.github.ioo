---
title: "How to scrape the ImageNet"
layout: post
---

### Result: 
**I wrote a [software tool](https://github.com/mf1024/ImageNet-datasets-downloader) which will prepare a dataset from ImageNet data using the URLs provided by ImageNet API.** 

**You can tell the tool: "I want a dataset with 200 classes with at least 800 images in each" and it will start collecting the images.**

**Also, in the process, I did a [little analysis](https://github.com/mf1024/data_analysis_notebooks/blob/master/imagenet_urls/ImageNet%20urls%20analysis.ipynb) of the state of the ImageNet image URLs.**

### The full story:

I needed to build and train a classification ConvNet on images that are larger than 32x32 pixels, so I had to find a dataset with bigger images labeled with classes. [ImageNet](http://image-net.org) is one such dataset. 

ImageNet is widely used for benchmarking image classification models. It contains 14 million images in more than 20 000 categories. 

One way to get the data would be to go for [the whole dataset](http://academictorrents.com/collection/imagenet-lsvrc-2015).

But I did not necessarily want nor need to download 150GB of data with images in every of the 20 000 classes. For prototyping and testing purposes smaller subsets of the dataset would be enough, for examplem, a dataset of 100 classes.

The ImageNet project does not own any of the images but they provide URL list for every image through the [API](http://image-net.org/download-API) [or in one file](http://image-net.org/download-imageurls).

I wanted to understand what kind of datasets from ImageNet I would be able to create. I parsed the URL list and plotted images-per-class histogram: 

![images_per_class](/img/imagenet/images_per_class.png)

The peak is at around 1200 images per class with more than 1000 such classes. Enough to create many variations of 100 class datasets of at least 1000 images per class. 

# The scraping and the first observations

So I wrote a program and started scraping.

**During the process I observed 5 things:**
1. Many of the sites are down. Many of the images cannot be found. (The URLs are from 2011)
2. Downloading images one by one each from its own URL is a slow process, but the URLs that are down and does not contain an image makes the process even slower - in most cases it's faster to download an image than to realize that the site won't respond with an image. 
3. There is a high diversity of sources of the images.
4. A big bulk of the images are from Flickr.
5. On some of the sites, if the image does not exist, another image is returned with some text that indicates that the image does not exist. For example:
![does_not_exist_1](/img/imagenet/does_not_exist_1.jpg)
Which is a problem not too easy to solve.

**I thought of an option which might solve some of the problems - to use just Flickr URLs.**

To continue I first checked if there are enough Flickr images in ImageNet for creating big enough datasets. **Quick URL parsing showed that 7 million of the images are from Flickr which is exactly half.**

Then I checked images per class with only Flickr URLs and I got the following image:
![images_per_class_flickr](/img/imagenet/images_per_class_flickr.png)

The peak is gone and the situation does not look that good anymore. 

To understand the situation a bit better I created reversed cumulative plot of the images per class, which instead of showing how many classes there are with exactly $$X$$ images per class it will show how many classes there are with at least $$X$$ images per class:
![images_per_class_flickr_culm](/img/imagenet/images_per_class_flickr_culm.png)

It shows that there are around 2000 classes with at least 1000 images per class which is still very good and more than enough for my purposes.

# Flickr URLs vs Other URLs

Now I need to check if using only Flickr URLs will improve the scraping process. 

**I ran my scraper for some time. In the process, I tried 25 000 random URLs. For every URL request I made, I marked if the image download was successful and I measured how much time it took to process the URL.**

First I wanted to understand how many of the URLs I tried were Flickr URLs and how many were other URLs:

![urls_encountered](/img/imagenet/urls_encountered.png)

The URLs are evenly distributed between Flick and other at least in my random sample, which will make it easier to compare them.

Here is a comparison of the time spent on requests for Flickr URLs vs other URLs:

![time_spent](/img/imagenet/time_spent.png)

Seems like the scraper is spending a lot of time on the other URLs. Let's see how productive is the time spent. Here is a comparison of successes from Flickr URLs vs other URLs:

![successes](/img/imagenet/successes.png)

Here we can that other URLs takes much more time and are less successful. I did some calculations on the averages: **approx. 80% of the Flickr URLs are successful where only 30% of other URLs are successful** 


And now let's check the most interesting metric - **how much time is spent per success** with Flickr URLs and other URLs:

![time_per_success](/img/imagenet/time_per_success.png)

The plot shows that the scraper on average spent 2 to 10 seconds per success on other URLs(the average is close to 4 seconds), while with the Flickr URLs the time per success consistently stays below 0.5 secs. A pretty significant difference. 0.5 seconds per image is still slow, but it's much faster and more consistent than using all of the URLs.

## The Imagenet Scraper

In the process, I wrote a [scraper](https://github.com/mf1024/ImageNet-datasets-downloader) which will create a dataset with $$Y$$ classes with $$X$$ images per class. I prepared it for use and put it on GitHub. 

The scraper will randomly pick classes with at least $$Y$$ images per class. But if you have any special requirements, you can specify a list of classes to download. To select the classes you can take a look at the [class list csv](https://github.com/mf1024/ImageNet-datasets-downloader/blob/master/classes_in_imagenet.csv) where I listed every class that appears in the Imagenet with its name, id, and the URL counts.

By default, the scraper will use only Flickr URLs, but if you are brave enough and ready to wait more and you are ready to clean up your data from bad images you can turn that option off.

