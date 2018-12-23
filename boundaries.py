import numpy as np
from scipy import fftpack

def wrap_boundary_liu(img, img_size):

    """
    Reducing boundary artifacts in image deconvolution
    Renting Liu, Jiaya Jia
    ICIP 2008
    """
    (H,W) = np.shape(img);
    H_w = img_size[0] - H;
    W_w = img_size[1] - W;

    #ret = np.zeros((img_size[0], img_size[1]));
    alpha = 1;
    HG = img[:,:];

    r_A = np.zeros((alpha*2+H_w,W));
    r_A[:alpha,:] = HG[-alpha:,:];
    r_A[-alpha:,:] = HG[:alpha,:];
    a = np.arange(H_w)/(H_w-1);
    #r_A(alpha+1:end-alpha, 1) = (1-a)*r_A(alpha,1) + a*r_A(end-alpha+1,1);
    r_A[alpha:-alpha,0] = (1-a)*r_A[alpha-1,0] + a*r_A[-alpha,0];
    #r_A(alpha+1:end-alpha, end) = (1-a)*r_A(alpha,end) + a*r_A(end-alpha+1,end);
    r_A[alpha:-alpha, -1] = (1-a)*r_A[alpha-1,-1] + a*r_A[-alpha,-1];
    
    r_B = np.zeros((H, alpha*2+W_w));
    r_B[:, :alpha] = HG[:, -alpha:];
    r_B[:, -alpha:] = HG[:, :alpha];
    a = np.arange(W_w)/(W_w-1);
    r_B[0, alpha:-alpha] = (1-a)*r_B[0,alpha-1] + a*r_B[0,-alpha];
    r_B[-1, alpha:-alpha] = (1-a)*r_B[-1,alpha-1] + a*r_B[-1,-alpha];
    
    if alpha == 1:
        A2 = solve_min_laplacian(r_A[alpha-1:,:]);
        B2 = solve_min_laplacian(r_B[:,alpha-1:]);
        r_A[alpha-1:,:] = A2;
        r_B[:,alpha-1:] = B2;
    else:
        A2 = solve_min_laplacian(r_A[alpha-1:-alpha+1,:]);
        r_A[alpha-1:-alpha+1,:] = A2;
        B2 = solve_min_laplacian(r_B[:,alpha-1:-alpha+1]);
        r_B[:,alpha-1:-alpha+1] = B2;
    A = r_A;
    B = r_B;

    r_C = np.zeros((alpha*2+H_w, alpha*2+W_w));
    r_C[:alpha, :] = B[-alpha:, :];
    r_C[-alpha:, :] = B[:alpha, :];
    r_C[:, :alpha] = A[:, -alpha:];
    r_C[:, -alpha:] = A[:, :alpha];

    if alpha == 1:
        C2 = C2 = solve_min_laplacian(r_C[alpha-1:, alpha-1:]);
        r_C[alpha-1:, alpha-1:] = C2;
    else:
        C2 = solve_min_laplacian(r_C[alpha-1:-alpha+1, alpha-1:-alpha+1]);
        r_C[alpha-1:-alpha+1, alpha-1:-alpha+1] = C2;
    C = r_C;
    #return C;
    A = A[alpha-1:-alpha-1, :];
    B = B[:, alpha:-alpha];
    C = C[alpha:-alpha, alpha:-alpha];
    ret = np.vstack((np.hstack((img,B)),np.hstack((A,C))));
    return ret;

def solve_min_laplacian(boundary_image):
    (H,W) = np.shape(boundary_image);

    # Laplacian
    f = np.zeros((H,W));
    # boundary image contains image intensities at boundaries
    boundary_image[1:-1, 1:-1] = 0;
    j = np.arange(2,H)-1;      
    k = np.arange(2,W)-1;      
    f_bp = np.zeros((H,W));
    f_bp[np.ix_(j,k)] = -4*boundary_image[np.ix_(j,k)] + boundary_image[np.ix_(j,k+1)] + boundary_image[np.ix_(j,k-1)] + boundary_image[np.ix_(j-1,k)] + boundary_image[np.ix_(j+1,k)];
    
    del(j,k);
    f1 = f - f_bp; # subtract boundary points contribution
    del(f_bp,f);

    # DST Sine Transform algo starts here
    f2 = f1[1:-1,1:-1];
    del(f1);

    # compute sine tranform
    if f2.shape[1] == 1:
        tt = fftpack.dst(f2,type=1,axis=0)/2;
    else:
        tt = fftpack.dst(f2,type=1)/2;
    
    if tt.shape[0] == 1:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt),type=1,axis=0)/2);
    else:
        f2sin = np.transpose(fftpack.dst(np.transpose(tt),type=1)/2);      
    del(f2)

    # compute Eigen Values
    [x,y] = np.meshgrid(np.arange(1,W-1), np.arange(1,H-1));
    denom = (2*np.cos(np.pi*x/(W-1))-2) + (2*np.cos(np.pi*y/(H-1)) - 2);
    
    # divide
    f3 = f2sin/denom;                          
    del(f2sin, x, y);

    # compute Inverse Sine Transform
    if f3.shape[0] == 1:
        tt = fftpack.idst(f3*2,type=1,axis=1)/(2*(f3.shape[1]+1));
    else:
        tt = fftpack.idst(f3*2,type=1,axis=0)/(2*(f3.shape[0]+1));
    del(f3);       
    if tt.shape[1] == 1:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2,type=1)/(2*(tt.shape[0]+1)));
    else:
        img_tt = np.transpose(fftpack.idst(np.transpose(tt)*2,type=1,axis=0)/(2*(tt.shape[1]+1)));
    del(tt);

    # put solution in inner points; outer points obtained from boundary image
    img_direct = boundary_image;
    img_direct[1:-1,1:-1] = 0;
    img_direct[1:-1,1:-1] = img_tt;
    return img_direct;