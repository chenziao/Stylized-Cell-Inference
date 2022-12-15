import numpy as np


def cart2pol(x, z):
    """
    Convert cartesian coordinates x,z to polar coordinate d,theta.
    theta is the angle between d and z-axis corresponding to rotation about y-axis.
    """
    x = np.asarray(x)
    z = np.asarray(z)
    d = np.sqrt(x ** 2 + z ** 2)
    theta = np.arctan2(x, z)
    return d, theta


def pol2cart(d, theta):
    """
    Convert polar coordinate d,theta to cartesian coordinates x,z.
    theta is the angle between d and z-axis corresponding to rotation about y-axis.
    """
    d = np.asarray(d)
    theta = np.asarray(theta)
    x = d * np.sin(theta)
    z = d * np.cos(theta)
    return x, z


def trivarnorm2unitsphere(x):
    """
    Project 3D coordinates onto unit sphere.
    x: n-by-3 array.
    Return n-by-3 array y.
    """
    x = np.asarray(x)
    shape = x.shape
    x = x.reshape((-1, 3))
    y = np.empty(x.shape)
    r = np.linalg.norm(x, axis=1)
    idx = r > 0
    y[idx, :] = x[idx, :] / r[idx, np.newaxis]
    y[~idx, :] = np.array([0., 1., 0.])
    y = y.reshape(shape)
    return y


def unitsphere2hphi(x):
    """
    Convert points on unit sphere to h, phi angle representation.
    x: n-by-3 array.
    Return n-by-2 array y.
    """
    x = np.asarray(x)
    shape = np.asarray(x.shape)
    shape[-1] = 2
    x = x.reshape((-1, 3))
    h = x[:, 1]
    phi = np.arctan2(x[:, 0], x[:, 2])
    y = np.stack((h, phi), axis=1)
    return y


def hphi2unitsphere(x):
    """
    Convert h, phi angle representation to points on unit sphere.
    x: n-by-2 array.
    Return n-by-3 array y.
    """
    x = np.asarray(x)
    shape = np.asarray(x.shape)
    shape[-1] = 3
    x = x.reshape((-1, 2))
    h = x[:, 0]
    phi = x[:, 1]
    y = np.empty((x.shape[0], 3))
    y[:, 1] = h
    r = np.sqrt(1 - h ** 2)
    y[:, 0] = r * np.sin(phi)
    y[:, 2] = r * np.cos(phi)
    y = y.reshape(shape)
    return y
