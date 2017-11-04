## Deep Learning paper and algorithm implementations

This repo collects exercises and provides code for deep learning papers and algorithms. Implementations are loosely organized by topic and grouped into folders. In addition to implementations, each folder contains a README with learning goals and a list of exercises. Both folders and exercises are arranged in increasing order of complexity.

All code is written in Python 3 and implementations are in either TensorFlow or PyTorch.

### Table of Contents

- 🙇 [Libraries: numpy, PyTorch, TensorFlow](0-libraries)
- 🎯 [Machine learning: linear algebra, non-deep classifiers](1-machine-learning)
- 🔑 [Neural net components: backprop, sigmoid, softmax, batchnorm, dropout](2-neural-nets)
- 📚 [Natural language processing, word2vec + subwords, NER, neural machine translation, attention](3-rnns)
- 🎨 [Image classification, convolutional networks, image segmentation, generative models](4-cnns)
- 💬 [Combined feature representations, VQA, captioning, saliency maps](5-rnns-cnns)

### Implemented

- Vanilla GAN [[code](simplest-gan)]
- VGG [[code](4-cnns/cnn.py)]
- Char-level RNN [[code](3-rnns/rnn.py)]
- Word2Vec [[code](2-neural-nets/word2vec.py)]
- Simple two-layer neural net [[code](2-neural-nets/two_layer_sigmoidal_net.py)]
- Numerical gradient checker [[code](2-neural-nets/gradient_checker.py)]
- Sigmoid [[code](2-neural-nets/sigmoid.py)]
- Softmax [[code](2-neural-nets/softmax.py)]
- Pytorch Exercises [[notebook](0-libraries/pytorch-exercises)]
- Kyubyong's numpy exercises [[notebook](0-libraries/numpy-exercises)]

### Resources

#### Classes:

- [fast.ai 1](http://course.fast.ai/): Practical Deep Learning For Coders
- [fast.ai 2](http://course.fast.ai/): Cutting Edge Deep Learning For Coders
- [fast.ai linalg](https://github.com/fastai/numerical-linear-algebra/blob/master/README.md): Computational Linear Algebra for Coders
- [CS224d](http://cs224d.stanford.edu/syllabus.html): Deep Learning for Natural Language Processing
- [CS231n](http://cs231n.stanford.edu/syllabus.html): Convolutional Neural Networks for Visual Recognition

#### Textbooks:

- [The Deep Learning Book](https://www.deeplearningbook.org/)

#### Collections of implementations:

- https://github.com/tensorflow/models
- https://github.com/dennybritz/models
- http://carpedm20.github.io

---

Format inspired by [Denny Britz](https://github.com/dennybritz/reinforcement-learning/blob/master/README.md) (my chief innovation on his format is that I added emojis).
