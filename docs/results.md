---
layout: page
title: Results
permalink: /results/
---

## GAN
### Introduction
After what we saw in the course, it was clear that, in order to generate images that could be perceived as closest to reality as possible, the GAN network would one of the best options. In this network, we train a discriminator and a generator. The discriminator tries to learn what a real image looks like, while the generator try to learn how to generate images that would be considered realistic. This means that if the discriminator is able to learn features that characterize what a real image should look like, such as smoothness, colours, shapes, etc. the generator should learn, eventually, how to fool the discriminator by making images that respects those features.

Since this is the goal of the project, i.e. generate realistic images, this seems to be the best choice for our network. By feeding the generator with a compressed vector of the initial inputs (the images with a hole of 32x32 pixels in the middle) with a noisy middle, the generator could learn to generate realistic images that fits the middle of the images. The discriminator could use the inner part of the images to try to determine whether the generated part is realistic with what a real image should look like.

### Methodology
First, we needed a way to encode the images. With that in mind, I trained an encoder to take the input (with its hole filled with random Gaussian noise) and compressed the information contained by the images in a vector of length 100. This network is a simple auto-encoder with a squared error cost function.

Once the training of this network was complete, it was then possible to train the GAN network. This was done by filling the input with random noise, encoding it, and feeding the vector that should now be an encoded version of the image to the generator. The discriminator could then try to figure out if this is a real or a fake image. The learning is done through back propagation (see Goodfellow 2017) using the Adam algorithm.

### Experiments
The GAN network seemed simple at first, but I had many difficulties with mine. First, I struggled with lots of instabilities that made my results highly unrealistic. My network was just generating random pixels and nothing else. The generator was not learning and the discriminator did not seem to accurately distinguish between real and false images. I did a lot of research on GAN and found that a lot of information. In the end, what really helped me was (Amos 2016). The website explains how to build a network that will fill holes in images using a DCGAN network and TensorFlow.

The trick that was given in (Amos 2016) to call the training function on the discriminator once and the training function for the generator twice was very helpful.

At first, my discriminator network was very small because it was said in many articles that it was easy to learn to discriminate images but very hard to learn how to generate some. This was a big mistake of mine. After making my discriminator bigger, it was clear that my network was learning a lot better.

### Results
After only 4 epochs of training, we can see that the generator learned to generate images that try to fit the colours and shapes around it. I was not able to run for more epochs because I had a lot of problems when trying to run my code on the server. Training with my CPU took a very long time, 3.5 hours to run over 1 epoch. There still seems to be some issue with the learning of the generator, though, since the gradient becomes very small after the first epoch. The result can be seen on the images that follows.

!["Figure 1: Examples of images generated by the generator for the test set"](https://raw.githubusercontent.com/GabrielBernard/Conditional_Image_Generation/master/docs/images/Resultats_gan.png "Figure 1: Generated images examples")

### Discussion

We can see that my model generates images, but that they are not really realistic. This is due to the incapacity of my generator to learn after a certain amount of epochs due to the discriminator being too good at its task, making the generator's network not receive enough gradient. The conclusion is still that a GAN network could give very good results, as proven by the project of (Paquette 2017) who generated some images that were very close to the source images with is DCGAN network. The only thing is that this network is highly sensitive, any slight error in its conception, such as a generator network that would be too small, and the risk of vanishing gradient for the generator is high.

Having to debug GPU errors on a distant server made me lose an awful lot of time on this project. Many of my attempts failed because of problems using the GPUs on the server, and I was not able to find solutions to all of those problems. This ultimately made me try to do the project on my personal computer which does not have a CUDA compatible GPU. This was a big problem for testing models since the learning was very long.

### Conclusion
My last model using GAN network was the most promising one, but due to time constraints and the fact that I was unable to properly use the server or a GPU to train my algorithm, the results were not as realistic as they could have been. Indeed (Amos (2016), Goodfellow (2017), Radford, Metz, and Chintala (2015)) and even (Paquette 2017) from the course, proved that it was feasible to achieve realistic completion of images using a DCGAN network. He even discussed how the sentences that comes with the images were not very useful in the end by generating nearly equally realistic images with fake captions.


## Learning to use Theano and Lasagne
My first attempt to solve the task was a simple convolutional encoder and decoder. Its goal was to try to learn a way to extract the features of the cropped data in order to compute the inner part. The cost function was a simple means squared error between the computed inner part and the real inner part of each image in the training set. Of course, this approach does not generalize well because the mean square error is not, in my opinion, a good way to measure the realism of a picture. The result were blurry, gray and have a problem of repeating pattern. I later found that this pattern was due to an error in the reshaping of the data. In any case, it was sure that this was not the network I could use to generate images considered realistic since it took a lot of training to obtain very poor results.

The results are given by the figure below.

!["Figure 1: First results"](https://raw.githubusercontent.com/GabrielBernard/Conditional_Image_Generation/master/docs/images/fig_blurry_patterned.png "Fig 1 first results")


## References
Amos, Brandon. 2016. “Image Completion with Deep Learning in Tensorflow.” [http://bamos.github.io/2016/08/09/deep-completion](http://bamos.github.io/2016/08/09/deep-completion).

Goodfellow, I. 2017. “NIPS 2016 Tutorial: Generative Adversarial Networks.” ArXiv E-Prints, December.

Paquette, Philip. 2017. “GAN.” Philip Paquette’s Blog. [https://ppaquette.github.io/](https://ppaquette.github.io/).

Radford, Alec, Luke Metz, and Soumith Chintala. 2015. “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.” CoRR abs/1511.06434. [http://arxiv.org/abs/1511.06434](http://arxiv.org/abs/1511.06434).