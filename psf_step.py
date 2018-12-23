import numpy as np
from utils import psf2otf,otf2psf,roll_zeropad
from scipy.signal import convolve2d
from scipy.optimize import fmin_cg

def gradJ(kf,x_dx,x_dy,y_dx,y_dy,gam,P_f,b):
    k_size = int(len(kf)**(0.5));
    k = kf.reshape(k_size,k_size);
    ret = -2*(b - otf2psf(P_f*psf2otf(k, x_dx.shape), (k_size,k_size)) - gam*k)
    return ret.flatten();

def J(kf,x_dx,x_dy,y_dx,y_dy,gam,P_f=0,b=0):
    k_size = int(len(kf)**(0.5));
    k = kf.reshape(k_size,k_size);
    return (np.linalg.norm(convolve2d(x_dx,k,mode='same',boundary='symm') + convolve2d(x_dy,k,mode='same',boundary='symm') - y_dx-y_dy)**2) + gam*(np.linalg.norm(k)**2);

def norm_k(k, mul=0.05):
    #ret = k*(k>=0);
    ret = k*(k >= np.max(k)*mul);
    ret = ret/np.sum(ret);
    return ret;

def psf_estim(x,y,gam,m,k0f):
    k_size = k0f.shape[0];
    (w,h) = x.shape;
    if len(k0f.shape) == 2:
        k0 = k0f.flatten();
    else:
        k0 = k0f;
    dx = np.array([[-1,1],[0,0]]);
    dy = np.array([[-1,0],[1,0]]);
    x_dx = convolve2d(x,dx,mode='valid');
    x_dy = convolve2d(x,dy,mode='valid')
    y_dx = convolve2d(y,dx,mode='valid');
    y_dy = convolve2d(y,dy,mode='valid')
    
    x_dx_f = np.fft.fft2(x_dx);
    x_dy_f = np.fft.fft2(x_dy);
    y_dx_f = np.fft.fft2(y_dx);
    y_dy_f = np.fft.fft2(y_dy);
    
    b = np.real(otf2psf(np.conj(x_dx_f)*y_dx_f + np.conj(x_dy_f)*y_dy_f, (k_size,k_size)));
    P_f = np.conj(x_dx_f)*x_dx_f + np.conj(x_dy_f)*x_dy_f;
    
    k = fmin_cg(J,k0,fprime=gradJ,args=(x_dx,x_dy,y_dx,y_dy,gam,P_f,b),gtol=1e-5,maxiter=m, disp=False);
    k = k.reshape(k_size,k_size);
    return norm_k(k);

def psf_center(psf):
    X,Y = np.meshgrid(np.arange(psf.shape[1]), np.arange(psf.shape[0]));
    xc1 = np.sum(psf*X);
    yc1 = np.sum(psf*Y);
    xc2 = (psf.shape[1]+1) / 2;
    yc2 = (psf.shape[0]+1) / 2;
    xshift = int(np.around(xc2 - xc1));
    yshift = int(np.around(yc2 - yc1));
    psf = roll_zeropad(psf,yshift,axis=0);
    psf = roll_zeropad(psf,xshift,axis=1);
    #psf = warpimage(psf, np.array([[1,0,-xshift],[0,1,-yshift]]));
    return psf;