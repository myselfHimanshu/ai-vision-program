# Assignment : Session 1

## What are Channels and Kernels?
Channel is a convolutional term used to refer to certain feature of an image. In practicality, an image from standard digital camera will have 3 channels (RGB (red, green, blue)). An image printed on a newspaper has 4 channels (CMYK). You can imagine three 2d matrices over each other, each having pixels values in range of [0,255].

![](https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png)

A grayscale image, has just one channel. The value of each pixel in the matrix ranges from 0 to 255 – zero indicating black and 255 indicating white.

| ![](https://upload.wikimedia.org/wikipedia/en/4/4c/Channel_digital_image_RGB_color.jpg) | ![](https://upload.wikimedia.org/wikipedia/en/4/45/Channel_digital_image_red.jpg) | <img src="https://upload.wikimedia.org/wikipedia/en/a/a8/Channel_digital_image_green.jpg" /> | ![](https://upload.wikimedia.org/wikipedia/en/b/b0/Channel_digital_image_blue.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RGB image                                                    | Red channel <br />(converted into grayscale)                 | Green channel<br />(converted into grayscale)                | Blue channel<br />(converted into grayscale)                 |

In above image, the red dress is much brighter in the red channel than in the other two, and the green part of the picture is shown much brighter in the green channel.

You shouldn't worry about RGB or CMYK specific channels. They are just like metrices which define an image. The main concept here is `channel`. An image can be divided into any number of channels, for an example, slide projector. You can imagine each slide as a channel and overlapping of these slides will form an image that gets projected over a screen.

Working on a specific color feature may be useful when you are asking a question to the network like, find me all yellow color flowers.

> A channel is set of relatable features.

><b>Channel</b> Synonym :<br>
> <i>feature map</i>, <i>convolved feature</i>, <i>activation map</i>

Let's take this image as an example to explain what a channel can be.

![](https://ak7.picdn.net/shutterstock/videos/23071327/thumb/12.jpg)

Now imagine that we have 26 channels for the given image. Let's say our channels are A-Z, as there are 26 alphabets so 26 channels.


- Channel A : where in all the <i>a</i> s from image are filtered out (wherever they are and however they are, just <i>a</i>s )
- Channel B : where in all the <i>b</i> s from image are filtered out.
- and so on, till Channel Z.


Let's talk about Channel A. Now a particular <i>a</i> in that channel is called `feature` , it can be big, small, tilted, anything but same feature. 

Now, when I asked you to filter out just <i>a</i> or extract just a single alphabet from the image to create a channel, you might need an extractor to do so. This extractor is termed as `kernel` . If you need to extract say <i>a</i>, you need this <i>c</i> kernel. 

> <b>Kernel</b> Synonym : <br>
> <i>feature extractor</i>,
> <i>n x n matrix</i>,
><i>filter</i>,
> <i>weights</i> 

> Each kernel gives us a channel. <br>
> What is a channel? <br>
> >Channel is a set of relatable features.

## Why should we (nearly) always use 3x3 kernels?

These kernels have values which are termed as <i>parameters</i>. We can say a 3x3 kernel will have 9 parameters.

Now coming to some mathematical concepts. 

- If we use a 5x5 kernel for convolution on 5x5 image, we will get output object of size 1x1. 
- If we use a 3x3 kernel for convolution on 5x5 image, we will get output object of size 3x3 and again convolving the output with 3x3 kernel will give us output object of size 1x1. 

So I can say, convolving with 5x5 kernel is same as convolving with 3x3 kernel (twice).

Now the question arises is, <i>What size kernel to use ? </i> 

The answer to above question lies in another question, 

<i>How many parameters we have to train on if we use 3x3 or 5x5 kernel?</i>

 Look at the table below which will tell you about total number of parameters to train if we use n x n size kernel.

| N <br />(N x N image) | total number of Parameters <br />( 3x3 kernel ) | total number of Parameters <br />( N x N kernel ) |
| :---------------------: | :-----------------------------------------------: | :-------------------------------------------------: |
|            5            |                   (3x3)*(2) = 18                   |                     (5x5) = 25                      |
|            7            |                   (3x3)*(3) = 27                   |                     (7x7) = 49                      |
|            9            |                   (3x3)*(4) = 36                   |                     (9x9) = 81                      |
|           11            |                   (3x3)*(5) = 45                   |                    (11x11) = 121                    |

The above table tells us why using 3x3 kernel is better than the other size kernels.

Less parameters means, model will take less time to train on. It will be faster. We need to build an CNN architecture with less parameters and that gives us best result.

There will be tradeoffs between parameters and result. But we need to come up with an elegant architecture and accordingly we will select our kernel size.

In many other blogs or research papers, you might see researchers using odd shaped kernel.

<i>Is it necessary to just use odd number shaped kernel?</i> 

It make sense to have our kernel odd shaped because it has axies of symmetry. 

Let's draw a black line on a piece of paper. If I want my machine to learn the difference between white line and black line. I need to tell it that left side of line is white and right side is also white and the middle column is black. The machine needs to differentiate the black pixels with what is not line i.e white pixels. We need to provide both the information. The machine needs to know the start and the end of a feature.

## How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

|image size|kernel size|output size|global receptive field size|
|--------|--------|--------|--------|
199x199|3x3|197x197|3x3|
197x197|3x3|195x195|5x5|
195x195|3x3|193x193|7x7|
193x193|3x3|191x191|9x9|
191x191|3x3|189x189|11x11|
189x189|3x3|187x187|13x13|
187x187|3x3|185x185|15x15|
185x185|3x3|183x183|17x17|
183x183|3x3|181x181|19x19|
181x181|3x3|179x179|21x21|
179x179|3x3|177x177|23x23|
177x177|3x3|175x175|25x25|
175x175|3x3|173x173|27x27|
173x173|3x3|171x171|29x29|
171x171|3x3|169x169|31x31|
169x169|3x3|167x167|33x33|
167x167|3x3|165x165|35x35|
165x165|3x3|163x163|37x37|
163x163|3x3|161x161|39x39|
161x161|3x3|159x159|41x41|
159x159|3x3|157x157|43x43|
157x157|3x3|155x155|45x45|
155x155|3x3|153x153|47x47|
153x153|3x3|151x151|49x49|
151x151|3x3|149x149|51x51|
149x149|3x3|147x147|53x53|
147x147|3x3|145x145|55x55|
145x145|3x3|143x143|57x57|
143x143|3x3|141x141|59x59|
141x141|3x3|139x139|61x61|
139x139|3x3|137x137|63x63|
137x137|3x3|135x135|65x65|
135x135|3x3|133x133|67x67|
133x133|3x3|131x131|69x69|
131x131|3x3|129x129|71x71|
129x129|3x3|127x127|73x73|
127x127|3x3|125x125|75x75|
125x125|3x3|123x123|77x77|
123x123|3x3|121x121|79x79|
121x121|3x3|119x119|81x81|
119x119|3x3|117x117|83x83|
117x117|3x3|115x115|85x85|
115x115|3x3|113x113|87x87|
113x113|3x3|111x111|89x89|
111x111|3x3|109x109|91x91|
109x109|3x3|107x107|93x93|
107x107|3x3|105x105|95x95|
105x105|3x3|103x103|97x97|
103x103|3x3|101x101|99x99|
101x101|3x3|99x99|101x101|
99x99|3x3|97x97|103x103|
97x97|3x3|95x95|105x105|
95x95|3x3|93x93|107x107|
93x93|3x3|91x91|109x109|
91x91|3x3|89x89|111x111|
89x89|3x3|87x87|113x113|
87x87|3x3|85x85|115x115|
85x85|3x3|83x83|117x117|
83x83|3x3|81x81|119x119|
81x81|3x3|79x79|121x121|
79x79|3x3|77x77|123x123|
77x77|3x3|75x75|125x125|
75x75|3x3|73x73|127x127|
73x73|3x3|71x71|129x129|
71x71|3x3|69x69|131x131|
69x69|3x3|67x67|133x133|
67x67|3x3|65x65|135x135|
65x65|3x3|63x63|137x137|
63x63|3x3|61x61|139x139|
61x61|3x3|59x59|141x141|
59x59|3x3|57x57|143x143|
57x57|3x3|55x55|145x145|
55x55|3x3|53x53|147x147|
53x53|3x3|51x51|149x149|
51x51|3x3|49x49|151x151|
49x49|3x3|47x47|153x153|
47x47|3x3|45x45|155x155|
45x45|3x3|43x43|157x157|
43x43|3x3|41x41|159x159|
41x41|3x3|39x39|161x161|
39x39|3x3|37x37|163x163|
37x37|3x3|35x35|165x165|
35x35|3x3|33x33|167x167|
33x33|3x3|31x31|169x169|
31x31|3x3|29x29|171x171|
29x29|3x3|27x27|173x173|
27x27|3x3|25x25|175x175|
25x25|3x3|23x23|177x177|
23x23|3x3|21x21|179x179|
21x21|3x3|19x19|181x181|
19x19|3x3|17x17|183x183|
17x17|3x3|15x15|185x185|
15x15|3x3|13x13|187x187|
13x13|3x3|11x11|189x189|
11x11|3x3|9x9|191x191|
9x9|3x3|7x7|193x193|
7x7|3x3|5x5|195x195|
5x5|3x3|3x3|197x197|
3x3|3x3|1x1|199x199|
Total layers used : 99

```python
"""
Assuming an image is a grayscale image with size (199x199)

formula to calculate output channel size for stride=1 and padding=0 is : 

n_out = (n_inp-k)+1
r_out = r_in + (k-1)

n : number of features
k : kernel size
r : receptive field size

The global receptive field of image is 1.

"""

n_in = 199 # input size of image 199x199
k = 3 # kernel size 3x3
r_in = 1 # global receptive field size for image

n_layers = 0 # number of layers

print("|image size|kernel size|output size|global receptive field size|")
print("|--------|--------|--------|--------|")
while ( n_in != 1):
    n_out = (n_in-k)+1 # output channel size, 197x197 for first layer
    r_out = r_in + (k-1) 
    print(f"{n_in}x{n_in}|{k}x{k}|{n_out}x{n_out}|{r_out}x{r_out}|")
    n_layers += 1
    n_in = n_out
    r_in = r_out

print(f"Total layers used : {n_layers}")

```

## How are kernels initialized? 

The values of a neural network must be initializes to random numbers.

Understand these 2 concepts :

<b>Deterministic Algorithms vs Non-Deterministic Algorithms</b>

Given an unordered array of numbers, a bubble sort algorithm will always execute in same way to give same ordered result.  

But some problems cannot be solved by this technique efficiently because of complexity of data. The algorithm may run but may never give you required solution or might run infinitely.

To solve such problems, we use non-deterministic algorithms.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Difference_between_deterministic_and_Nondeterministic.svg/950px-Difference_between_deterministic_and_Nondeterministic.svg.png)

A deterministic algorithm that performs f(n) steps always finishes in f(n) steps and always returns the same result. A non deterministic algorithm that has f(n) levels might not return the same result on different runs. A non deterministic algorithm may never finish due to the potentially infinite size of the fixed height tree.

These non-deterministic algorithm will arive at approximate solution but will be fast. These solution will often be satisfactory for such problems.

These kind of algorithms make use of randomness. You might have studied about gradient descent algorithm, these are referred to as [stochastic algorithms](https://en.wikipedia.org/wiki/Stochastic_optimization).

The process of finding solution is incremental, starting from a point in sapce of possible solutions to good enough solution. As we know nothing about space, we start with random chosen point.

Neural Networks are trained using these kind of algorithms. 

> Training algorithms for deep learning models are usually iterative in nature and thus require the user to specify some initial point from which to begin the iterations. Moreover, training deep models is a sufficiently difficult task that most algorithms are strongly affected by the choice of initialization.<br>
> Perhaps the only property known with complete certainty is that the initial parameters need to “break symmetry” between diﬀerent units. If two hidden units with the same activation function are connected to the same inputs, then these units must have diﬀerent initial parameters. If they have the same initial parameters, then a deterministic learning algorithm applied to a deterministic costant model will constantly update both of these units in the same way.
- page 296,297, [deep learning book](https://www.deeplearningbook.org/contents/optimization.html).

A careful initialization of the network can speed up the learning process.

## What happens during the training of a DNN?

<img src="https://latex.codecogs.com/svg.latex?\Large&space;f^*" title="\Large f^*" />

The goal of a feedforward network is to approximate some function <img src="https://latex.codecogs.com/svg.latex?\Large&space;f^*" title="\Large f^*" />. For a classiﬁer, <img src="https://latex.codecogs.com/svg.latex?\Large&space;y=f^*(x)" title="\Large y=f^*(x)" /> maps an input x to a category y.

We have our input images which are feeded into a network, these networks are ment to find the weights at each layer of a neural network, where in each layers output is calculated with some compute function of weights and output of previous layer.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image12.png)

These weights are initialized randomly at first.

First layer's task is to find edges and gradients, then second layer's task is to find textures in patterns which are formed by the combination of features of its previous layer and another set of weights and so on. Each of the layers output is passed to its next layer upto till prediction layer. 

Then the network compares the result of predicted layer with the expected outputs and calculate loss with them. This loss is backpropagated to the network, where in the network adjust its weight with gradient descent. This process of forward feeding of the outputs, backpropagating of calucated loss and updation of parameters goes on till the loss keeps on reducing.



