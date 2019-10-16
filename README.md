# Harris-corner-detection

### Description:
Implement Harris-corner-detection from scratch without using functions that can get the result directly.

### Packages used:
1. numpy: array operation
2. cv2: load, save, show image
3. math: generate some special arithmetic operators
4. matlplotlib: save, show image
5. scipy: convolution, generate filters

### Function:
1. gaussian_smooth:
```
    description:
        convolve image with gaussian filter to generate smoothed image
    input: 
        img: image(H * W * 3)
        kernel_size: gaussian filter kernel size(N * N)
        sigma: gaussian filter variance
    output: 
        img_smooth: image smoothed by gaussian filter(H * W * 3)
```
2. sobel_edge_detection:
```
    description:    
        convolve image sobel filter to generate gradient information of image
    input: 
        img: gray scale of smoothed image(H * W)
    output: 
        grad_mag: gradient magnitude(H * W)
        grad_dir: gradient direction(H * W)
        img_gx, img_gy: x and y direction of image gradient(H * W)
```
3. structure_tensor:
```
    description:    
        generate the elements of structure tensor
    input:
        ix, iy: x and y direction of image gradient(H * W)      
    output:
        ixx, ixy, iyy: three kinds of elements in structure tensor(H * W)
```        
4. nms: 
```    
    description:
        calculate corner response of each matrix and select pixels with response 
        that larger than threshold as corners
    input:
        ix, iy: x and y direction of image gradient(H * W)
        win_size: window size of structure tensor(N * N)
        k: empirical constant of corner response
        local_size: window size to find local maximum(N * N)
        sigma: gaussian filter variance
    output:
        R: corner response(H * W)
        corner: selected corners(H * W) 
```        
5. rotate:
```
    description:    
        rotate image
    input:
        img: image(H * W * 3)
        angle: rotation angle
        center: rotation center((w,h))
        scale: relative size to input image
    output:
        rotated: rotated image(H * W * 3)
```
6. resize:
```
    description:
        resize image
    input:
        img: image(H * W * 3)
        scale: relative size to input image
        inter: type of interpolation(cv2 built-in parameter)
    output:
        resized: resized image(H * W * 3)
```
### Parameters:
    kernel_size: kernel size of Guassian blur
    win_size: window size of structure tensor
    local_size: window size of finding local maximum in nms
    ROTATE: whether to rotate
    RESIZE: whether to resize

### Procedure:
1. read image    
2. rotate/scale image(optional)
3. smooth image
4. convert smoothed image to gray-scale
5. sobel edge detection
6. generate colour gradient direction image
```
    description:
        First, use a threshold to filter out pixels with low gradient magnitude.
        Second, generate two masks that store the position of the positive and 
        negative gradient direction respectively.Third, interpolate negative 
        gradient direction to [0.1,0.5] and positive one to [0.6,1.0]. Last, 
        product these two results with corresponding masks and sum up to generate
        colour gradient direction image.
```		
7. non-maximum suppression
