import numpy as np
import matplotlib.pyplot as plt
import math

def whiten_background(image):
    Image = image.copy();
    Image = Image-Image.min();
    Image = Image/np.mean(Image);
    Image = np.clip(Image, 0, 1);
    return Image;

def show_images(input, input_title, output, output_title, typo='vertical',size=(20,40),axis='false'):
    if typo == 'vertical':
        fig, (im_input, im_output) = plt.subplots(2, 1, figsize=size)
    else:
        fig, (im_input, im_output) = plt.subplots(1, 2, figsize=size)
    im_input.imshow(input, cmap='gray')
    im_input.set_title(input_title)
    im_output.imshow(output, cmap='gray')
    im_output.set_title(output_title)
    if axis == 'false':
        im_input.set_axis_off()
        im_output.set_axis_off()
    fig.show()
    
def expand_k(k,shape):
    ret = np.zeros(shape);
    ret[:k.shape[0],:k.shape[1]] = k;
    return ret;
    
def zero_pad(image, shape, position='corner'):
    """
    Extends image to a certain size with zeros

    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered

    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image

    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img

def psf2otf(psf, shape):
    """
    Convert point-spread function to optical transfer function.

    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.

    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.

    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf

def length(a):
    return np.max(a.shape);

def otf2psf(otf,outSize):
    """
    OTF2PSF Convert optical transfer function to point-spread function.
    PSF = OTF2PSF(OTF) computes the inverse Fast Fourier Transform (IFFT)
    of the optical transfer function (OTF) array and creates a point spread
    function (PSF), centered at the origin. By default, the PSF is the same 
    size as the OTF.

    PSF = OTF2PSF(OTF,OUTSIZE) converts the OTF array into a PSF array of
    specified size OUTSIZE. The OUTSIZE must not exceed the size of the
    OTF array in any dimension.

    To center the PSF at the origin, OTF2PSF circularly shifts the values
    of the output array down (or to the right) until the (1,1) element
    reaches the central position, then it crops the result to match
    dimensions specified by OUTSIZE.

    Note that this function is used in image convolution/deconvolution 
    when the operations involve the FFT. 

    Class Support
    -------------
    OTF can be any nonsparse, numeric array. PSF is of class double. 

    Adapted from MATLAB otf2psf function  
    """
    eps = 2.2204e-16;
    if np.sum(otf != 0) == 0:
        return np.zeros(shape);
    else:
        psf = np.fft.ifft2(otf);

    # Estimate the rough number of operations involved in the 
    # computation of the IFFT 
    otfSize = otf.shape;
    nElem = np.prod(otfSize);
    nOps  = 0;
    for k in range(len(otfSize)):
        nffts = nElem/otfSize[k];
        nOps  = nOps + otfSize[k]*np.log2(otfSize[k])*nffts;

   # Discard the imaginary part of the psf if it's within roundoff error.
    if np.max(np.abs(np.imag(psf)))/np.max(np.abs(psf)) <= nOps*eps:
        psf = np.real(psf);

   # Circularly shift psf so that (1,1) element is moved to the
   # appropriate center position.
    psf = np.roll(np.roll(psf,math.floor(outSize[0]/2),axis=0),math.floor(outSize[1]/2),axis=1);
   # Crop output array.
    return psf[:outSize[0],:outSize[1]];

def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.
    Adapted from https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding
    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
def l0_norm(img):
    tmp = (img != 0);
    return np.sum(tmp);

def P(img,sigma):
    return sigma*l0_norm(img) + l0_norm(im_grad(img));

def laplace(img,bc='reflexive'):
    return shift(img*1.0,[-1,0],bc) + shift(img*1.0,[1,0],bc) - 4.0*img + shift(img*1.0,[0,-1],bc) + shift(img*1.0,[0,1],bc);

def shock_filter(img, dt=1, n=1):
    I = img*1.0;
    if len(img.shape) == 2:
        for i in range(n):
            I = I - np.sign(laplace(I))*np.linalg.norm(im_grad(I),axis=2)*dt;
    else:
        for i in range(n):
            I = I - np.sign(laplace(I))*np.linalg.norm(im_grad(I),axis=3)*dt;
    return I;

def grad_thresh(img,m):
    r = 0;
    return 0;

def kern_norm(k, threshold=20):
    #ret = k[:];
    m = np.max(k)/threshold;
    #for i in range(k.shape[0]):
    #    for i in range(k.shape[1]):
    #        if k[i,j] < m:
    #            ret[i,j] = 0;
    ret = k*(k>m);
    return ret/np.sum(ret);


def myconvolve(image,kernel0):
    if len(kernel0.shape) == 1:
        kernel = kernel0.reshape(int(len(kernel0)**(0.5)),int(len(kernel0)**(0.5)));
    else:
        kernel = kernel0;
    if len(image.shape) == 2:
        return np.fft.ifft2(np.fft.fft2(image)*psf2otf(kernel,image.shape)).real;
        #return convolve2d(image,kernel,mode='same',boundary='wrap');
    else:
        #return np.stack((convolve2d(image[:,:,0],kernel,mode='same',boundary='wrap'),convolve2d(image[:,:,1],kernel,mode='same',boundary='wrap')),axis=-1);
        return np.stack((np.fft.ifft2(np.fft.fft2(image[:,:,0])*psf2otf(kernel,image[:,:,0].shape)).real,np.fft.ifft2(np.fft.fft2(image[:,:,1])*psf2otf(kernel,image[:,:,0].shape)).real),axis=-1);

def pad_size(image,k_size):
    w = image.shape[0];
    h = image.shape[1];
    d2 = int(math.log(w + k_size + 1,2))+1;
    d3 = int(math.log(w + k_size + 1,3))+1;
    d5 = int(math.log(w + k_size + 1,5))+1;
    d7 = int(math.log(w + k_size + 1,7))+1;
    wn = np.min((2**d2,3**d3,5**d5,7**d7));
    d2 = int(math.log(h + k_size + 1,2))+1;
    d3 = int(math.log(h + k_size + 1,3))+1;
    d5 = int(math.log(h + k_size + 1,5))+1;
    d7 = int(math.log(h + k_size + 1,7))+1;
    hn = np.min((2**d2,3**d3,5**d5,7**d7));
    return (wn,hn);

def unpad(image,dims):
    return image[:dims[0],:dims[1]];