import numpy as np
from latent_step import latent_image, im_normalize
from psf_step import psf_estim, psf_center
from utils import whiten_background
from cv2 import connectedComponents
from final_deconv import ringing_artifacts_removal
from skimage import img_as_ubyte
from skimage import io
from PIL import Image
from tqdm import tqdm

def blind_deconv(img,k,num_iter,lamb, sigma, gam, beta_max, mu_max, wei_grad,lamb_tv,lamb_l0):
    img_max = np.max(img);
    img_min = np.min(img);
    
    for i in tqdm(range(num_iter)):
        lat = latent_image(img,k,lamb, sigma, beta_max, mu_max, wei_grad);
        lat = im_normalize(lat,img_min,img_max);
#        lat = whiten_background(lat);
        k_p = k[:]*1.0;
        k = psf_estim(lat, img, gam, 30, k_p);
        
        a,label = connectedComponents(img_as_ubyte(k), connectivity=8);
        for j in range(0,a):
            label_sum = np.sum(k[label == j]);
            if label_sum < 0.1: 
                k[label == j] = 0;
        k=k*(k>0);
        k=k/np.sum(k);
        
        if lamb != 0:
            lamb = np.max((lamb/1.1, 1e-4));
        lamb = lamb/1.1;
        if wei_grad!=0:
            wei_grad = np.max((wei_grad/1.1, 1e-4));
        im = Image.fromarray(img_as_ubyte(k/np.max(k)));
        im.save('temp_k.png');
        io.imsave('temp_lat.png',im_normalize(lat,img_min,img_max));
    k = psf_center(k);
    lat = ringing_artifacts_removal(img, k, lamb_tv, lamb_l0, wei_grad);
    return lat, k;