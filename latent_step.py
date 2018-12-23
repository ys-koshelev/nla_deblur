import numpy as np
from utils import psf2otf,otf2psf,pad_size
from boundaries import wrap_boundary_liu

def im_normalize(x,x_min,x_max):
    #x = x - np.min(x) + x_min;
    #x = x/(x.max()-x.min())*(x_max-x_min);
    #x = x - np.min(x) + x_min;
    ret = x.copy();
    ret = ret - ret.min();
    ret = ret/ret.max();
    return ret;

def compute_u(x,lamb,sigma,beta):
    ret = x*(np.abs(x)**2 >= lamb*sigma/beta);
    return ret;

def compute_g(h,v,lamb,mu):
    mask = ((h**2) + (v**2)) >= lamb/mu;
    return np.stack((h,v),axis=-1)*np.stack((mask,mask),axis=-1);

def closed_form(h,v,term1,denom,u,mu,beta):
    term2 = np.hstack((np.expand_dims(h[:,-1] - h[:, 0],axis=1), -1.0*np.diff(h,1,1)));
    term2 = term2 + np.vstack((v[-1,:] - v[0, :], -1.0*np.diff(v,1,0)));
    ret = (term1 + mu*np.fft.fft2(term2) + beta*np.fft.fft2(u))/(denom);
    return np.fft.ifft2(ret).real

def latent_image(y,k,lamb,sigma,beta_max,mu_max,wei_grad,kappa=2):
    x = wrap_boundary_liu(y,pad_size(y,k.shape[0]));
    dx = np.array([[0.0,0.0,0.0],[1.0,-1.0,0.0],[0.0,0.0,0.0]]);
    dy = np.array([[1.0], [-1.0]]);
    (N,M) = x.shape;
    dx_f = psf2otf(dx,(N,M));
    dy_f = psf2otf(dy,(N,M));
    k_f = psf2otf(k,(N,M));
    k_f_norm = np.abs(k_f)**2;
    grad_norm = (np.abs(dx_f)**2) + (np.abs(dy_f)**2);
    term1 = np.conj(k_f)*np.fft.fft2(x);
    beta = 2*lamb;
    while beta < beta_max:
        u = compute_u(x,lamb,sigma,beta);
        mu = 2*wei_grad;
        while mu < mu_max:
            denom = k_f_norm + mu*grad_norm + beta;
            h = np.hstack((np.diff(x,1,1)*1.0, np.expand_dims(x[:,0] - x[:,-1],axis=1)));
            v = np.vstack((np.diff(x,1,0), x[0,:] - x[-1,:]));
            g = compute_g(h,v,wei_grad,mu);
            x = closed_form(g[:,:,0],g[:,:,1],term1,denom,u,mu,beta);
            mu = kappa*mu;
            if wei_grad == 0:
                break;
        beta = kappa*beta;
    ret = x[:y.shape[0],:y.shape[1]];
#    ret = im_normalize(ret,y.min(),y.max());
    return ret;


def latent_image_dering(y,k,lamb,kappa=2):
    x = wrap_boundary_liu(y,pad_size(y,k.shape[0]));
    betamax = 1e5;
    dx = np.array([[0.0,0.0,0.0],[1.0,-1.0,0.0],[0.0,0.0,0.0]]);
    dy = np.array([[1.0], [-1.0]]);
    (N,M) = x.shape;
    dx_f = psf2otf(dx,(N,M));
    dy_f = psf2otf(dy,(N,M));
    k_f = psf2otf(k,(N,M));
    k_f_norm = np.abs(k_f)**2;
    grad_norm = (np.abs(dx_f)**2) + (np.abs(dy_f)**2);
    term1 = np.conj(k_f)*np.fft.fft2(x);
    beta = 2*lamb;
    while beta < betamax:
        denom = k_f_norm + beta*grad_norm;
        h = np.hstack((np.diff(x,1,1)*1.0, np.expand_dims(x[:,0] - x[:,-1],axis=1)));
        v = np.vstack((np.diff(x,1,0), x[0,:] - x[-1,:]));
        g = compute_g(h,v,lamb,beta);
        x = closed_form(g[:,:,0],g[:,:,1],term1,denom,x,beta,0);
        beta = kappa*beta;
    ret = x[:y.shape[0],:y.shape[1]];
    ret = im_normalize(ret,y.min(),y.max());
    return ret;

def synth_kernel(n):
    k = np.zeros((n,n));
    k[int((n - 1)/2-1),int((n - 1)/2-1):int((n - 1)/2+1)] = 0.5;
    return k;