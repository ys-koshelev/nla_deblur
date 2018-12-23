from deblur import blind_deconv
from latent_step import synth_kernel,im_normalize
from utils import whiten_background
from argparse import ArgumentParser
from numpy import load,loadtxt,savetxt
from skimage import io
import sys

def main():
    # parsing arguments
    parser = ArgumentParser(description='Applying blind deconvolution')

    parser.add_argument('--lambda_', '-lam', default=4e-3, type=float,
                        help='Lagrange multiplier lambda for image prior. Default: 4e-3.');
    parser.add_argument('--gamma', '-gam', default=2560, type=float,
                        help='Lagrange multiplier gamma for kernel prior. Default: 2560.');
    parser.add_argument('--sigma', '-s', default=1, type=float,
                        help='Weight sigma for ||x|| in image prior P(x) = sigma||x||_0 + ||grad(x)||_0. Default: 1.');
    parser.add_argument('--beta_max', '-bmax', default=2**3, type=float,
                        help='Maximum value for iteration over beta. Default: 2**3.');
    parser.add_argument('--mu_max', '-mmax', default=1e5, type=float,
                        help='Maximum value for iteration over mu. Default: 1e5.');
    parser.add_argument('--wei_grad', '-wgrad', default=4e-3, type=float,
                        help='Initial value for mu. Default: 4e3.');
    parser.add_argument('--lambda_tv', '-lamtv', default=0.003, type=float,
                        help='Lagrange multiplier lambda for image prior for last deblur iteration (TV). Default: 0.003.');
    parser.add_argument('--lambda_l0', '-laml0', default=1e-3, type=float,
                        help='Lagrange multiplier lambda for image prior for last deblur iteration (L0). Default: 1e-3.');
    parser.add_argument('--image_path', '-img', default='test.tif',
                        help='Path to image to deblur. Default: test.tif.');
    parser.add_argument('--kernel_size', '-ksize', default=25, type=int,
                        help='PSF size in px. Default: 25.');
    parser.add_argument('--num_iter', '-iter', default=50, type=int,
                        help='Number of iterations to produce. Default: 50.');
    args = parser.parse_args()

    # loading data
    image = io.imread(args.image_path)/255.0;
    inference, k, = blind_deconv(image[10:-15, 10:-5],
                                 synth_kernel(args.kernel_size),
                                 args.num_iter,
                                 args.lambda_,
                                 args.sigma,
                                 args.gamma,
                                 args.beta_max,
                                 args.mu_max,
                                 args.wei_grad,
                                 args.lambda_tv,
                                 args.lambda_l0);
                                 
    io.imsave('deblurred.png', whiten_background(inference));
    io.imsave('kernel.png', im_normalize(k, 0, 1));
    return inference, k;

if __name__ == "__main__":
    sys.exit(main());
