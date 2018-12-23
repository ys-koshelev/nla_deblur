import numpy as np
from utils import psf2otf,otf2psf,pad_size
from boundaries import wrap_boundary_liu
from latent_step import latent_image_dering

def ringing_artifacts_removal(y, kernel, lambda_tv, lambda_l0, weight_ring):
    (H,W) = y.shape;
#    y_pad = y.copy();
    y_pad = wrap_boundary_liu(y, pad_size(y,kernel.shape[0]));
    Latent_tv = deblurring_adm_aniso(y_pad, kernel, lambda_tv);
    Latent_tv = Latent_tv[:H,:W];
    if weight_ring==0:
        return Latent_tv;
    #Latent_l0 = latent_image(y_pad, kernel, lambda_l0,0,beta_max,mu_max,wei_grad);
    Latent_l0 = latent_image_dering(y_pad, kernel, lambda_l0,2);
    Latent_l0 = Latent_l0[:H,:W];
    diff = Latent_tv - Latent_l0;
    bf_diff = bilateral_filter(diff, 3, 0.1);
    return Latent_tv - weight_ring*bf_diff;

def deblurring_adm_aniso(B, k, lamb):
    beta = 1/lamb;
    beta_rate = 2*(2**0.5);
    beta_min = 0.001;
    (m,n) = B.shape; 
    I = B; 
    Nomin1, Denom1, Denom2 = computeDenominator(B, k);
    
    Ix = np.hstack((np.diff(I,1,1), np.expand_dims(I[:,0] - I[:,-1],axis=1)));
    Iy = np.vstack((np.diff(I,1,0), I[0,:] - I[-1,:]));
       
    while beta > beta_min:
        gamma = 1/(2*beta);
        Denom = Denom1 + gamma*Denom2;
        
        a = np.abs(Ix) - beta*lamb;
        Wx = a*(a>0)*np.sign(Ix);
        a = np.abs(Iy) - beta*lamb;
        Wy = a*(a>0)*np.sign(Iy);
        Wxx = np.hstack((np.expand_dims(Wx[:,n-1] - Wx[:, 0],axis=1), -np.diff(Wx,1,1)));
        Wxx = Wxx + np.vstack((Wy[m-1,:] - Wy[0, :], -np.diff(Wy,1,0)));
        
        
        Fyout = (Nomin1 + gamma*np.fft.fft2(Wxx))/Denom; 
        
        I = (np.fft.ifft2(Fyout)).real;
        
        Ix = np.hstack((np.diff(I,1,1), np.expand_dims(I[:,0] - I[:,n-1],axis=1)));
        Iy = np.vstack((np.diff(I,1,0), I[0,:] - I[m-1,:]));
        beta = beta/2;
    return I;

def computeDenominator(y, k):
    sizey = y.shape;
    otfk  = psf2otf(k, sizey); 
    Nomin1 = np.conj(otfk)*np.fft.fft2(y);
    Denom1 = np.abs(otfk)**2;
    dx = np.array([[0.0,0.0,0.0],[1.0,-1.0,0.0],[0.0,0.0,0.0]]);
    dy = np.array([[1.0], [-1.0]]);
    Denom2 = np.abs(psf2otf(dx,sizey))**2 + np.abs(psf2otf(dy,sizey))**2;
    return Nomin1, Denom1, Denom2;

def matlab_style_gauss2D(dim,sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    shape = (dim,dim);
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def bilateral_filter(img, sigma_s, sigma, boundary_method='edge', s_size = 0):
    if isinstance(img[0,0],int) == 1: 
        img = single(img)/255.0;
    (h,w) = img.shape;

    lab = img;    

    if s_size !=0:
        fr = s_size;
    else:
        fr = int(np.ceil(sigma_s*3));

    p_img = np.pad(img, fr, mode=boundary_method);
    p_lab = np.pad(lab, fr, mode=boundary_method);

    u = fr+1; 
    b = u+h-1;
    l = fr+1;
    r = l+w-1;

    r_img = np.zeros((h, w));
    w_sum = np.zeros((h, w));

    spatial_weight = matlab_style_gauss2D(2*fr+1, sigma_s);
    ss = sigma * sigma;

    for y in range(-fr,fr+1):
        for x in range(-fr,fr+1):
            w_s = spatial_weight[y+fr, x+fr];
            n_img = p_img[u+y-1:b+y, l+x-1:r+x];
            n_lab = p_lab[u+y-1:b+y, l+x-1:r+x];
            f_diff = lab - n_lab;
            f_dist = f_diff**2;
            w_f = np.exp(-0.5*(f_dist/ss));
            w_t = w_s* w_f;
            r_img = r_img + n_img*w_t;
            w_sum = w_sum + w_t;

    r_img = r_img/w_sum;
    return r_img;