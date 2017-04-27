---
layout: page
title: Results
permalink: /results/
---

## GAN
### Introduction
After what we saw in the course, it was clear that, in order to generate images that could be perceived as closest to reality as possible, the GAN network would be the best option. In this network, we train a discriminator and a generator. The discriminator tries to learn what a real image looks like, while the generator try to learn how to generate images that would be considered realistic. This means that if the discriminator is able to learn features that characterize what a real image should look like, such as smoothness, colours, shapes, etc. the generator should learn, eventually, how to fool the discriminator by making images that respects those features.

Since this is the goal of the project, i.e. generate realistic images, this seems to be the best choice for our network. By feeding the generator with a compressed vector of the initial inputs (the images with a hole of 32x32 pixels in the middle) with a noisy middle, the generator could learn to generate realistic images that fits the middle of the images. The discriminator could use the inner part of the images to try to determine whether the generated part is realistic with what a real image should look like.

### Methodology
First, we needed a way to encode the images. With that in mind, I trained an encoder to take the input (with its hole filled with random Gaussian noise) and compressed the information contained by the images in a vector of length 100. This network is a simple auto-encoder with a cost function using a binary cross entropy.

Once the training of this network was complete, it was then possible to train the GAN network. This was done by filling the input with random noise, encoding it, and feeding the vector encoding the image to the generator. The discriminator could then try to figure if this was a real or a fake image. The learning is done through back propagation [see @2017arXiv170100160G].

### Results

[comment]: <> (![alt text](https://github.com/GabrielBernard/Conditional_Image_Generation/tree/master/docs/images/fig_blurry_patterned.png "Figure 1: Blurry, gray and repeated pattern in predicted images."))


## Learning to use Theano and Lasagne
My first attempt to solve the task was a simple convolutional encoder and decoder. It's goal was to try to learn a way to extract the features of the cropped data in order to compute the inner part. The cost function was a simple mean squared error between the computed inner part and the real inner part of each image in the training set. Of course, this approach does not generalize well because the mean square error is not, in my opinion, a good way to measure the realism of a picture. The result where blurry, gray and have a problem of repeating pattern to which I could not find the source. These results are given by the figure bellow.

!["Figure 1: Blurry, gray and repeated pattern in predicted images."](../images/fig_blurry_patterned.png "Figure 1: Blurry, gray and repeated pattern in predicted images.")
