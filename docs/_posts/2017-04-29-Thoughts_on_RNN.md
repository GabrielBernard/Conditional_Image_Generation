---
layout: post
title:  "Thoughts on RNN"
date:   2017-04-29 14:15:00 -0500
categories: info
---

For the course project, a set of sentences describing each image was provided in the dataset. We were supposed to try to find a way to use those sentences to help the generative model in its task. I think the use of an RNN would have made this possible.

In fact, building an LSTM network that would have tried to find useful keywords in the sentences in order to associate them with features in the real image could have been used. Whether if it would have a meaningful impact on the image generation is another matter, but that is what I would have tested if my DCGAN network would have worked sooner and with better results.
