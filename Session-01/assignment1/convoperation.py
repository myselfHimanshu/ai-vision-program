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
