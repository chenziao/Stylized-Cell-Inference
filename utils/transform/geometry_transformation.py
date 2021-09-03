import numpy as np

def cart2pol(x, z):
    """
    Convert cartesian coordinates x,z to polar coordinate d,theta.
    theta is the angle between d and z-axis corresponding to rotation about y-axis.
    """
    d = np.sqrt(x**2 + z**2)
    theta = np.arctan2(x, z)
    return d, theta

def pol2cart(d, theta):
    """
    Convert polar coordinate d,theta to cartesian coordinates x,z.
    theta is the angle between d and z-axis corresponding to rotation about y-axis.
    """
    x = d * np.sin(theta)
    z = d * np.cos(theta)
    return x, z

def trivarnorm2unitsphere(x):
    """
    Project 3D coordinates onto unit sphere.
    x: n-by-3 array.
    Return n-by-3 array y.
    """
    x = x.reshape((-1,3))
    y = np.empty(x.shape)
    r = np.linalg.norm(x,axis=1)
    idx = r>0
    y[idx,:] = x[idx,:]/np.expand_dims(r[idx],axis=1)
    y[~idx,:] = np.array([0.,1.,0.])
    return y

def unitsphere2hphi(x):
    """Convert points on unit sphere to h, phi angle representation."""
    x = x.reshape((-1,3))
    h = x[:,1]
    phi = np.arctan2(x[:,0],x[:,2])
    return h, phi
