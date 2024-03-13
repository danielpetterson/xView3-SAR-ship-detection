import errno
import math
import os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

########################################################################################
################################ XVIEW 3 SPECIFIC UTILS ################################
########################################################################################


def pad(vh, rows, cols):
    """
    Pad an image to make it divisible by some block_size.
    Pad on the right and bottom edges so annotations are still usable.
    """
    r, c = vh.shape
    to_rows = math.ceil(r / rows) * rows
    to_cols = math.ceil(c / cols) * cols
    pad_rows = to_rows - r
    pad_cols = to_cols - c
    vh_pad = np.pad(
        vh, pad_width=((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0
    )
    return vh_pad, pad_rows, pad_cols


def chip_sar_img(input_img, sz):
    """
    Takes a raster from xView3 as input and outputs
    a set of chips and the coordinate grid for a
    given chip size

    Args:
        input_img (numpy.array): Input image in np.array form
        sz (int): Size of chip (will be sz x sz x # of channlls)

    Returns:
        images: set of image chips
        images_grid: grid coordinates for each chip
    """
    # The input_img is presumed to already be padded
    images = view_as_blocks(input_img, (sz, sz))
    images_grid = images.reshape(
        int(input_img.shape[0] / sz), int(input_img.shape[1] / sz), sz, sz
    )
    return images, images_grid


def view_as_blocks(arr, block_size):
    """
    Break up an image into blocks and return array.
    """
    m, n = arr.shape
    M, N = block_size
    return arr.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M, N)


def find_nearest(lon, lat, x, y):
    """Find nearest row/col pixel for x/y coordinate.
    lon, lat : 2D image
    x, y : scalar coordinates
    """
    X = np.abs(lon - x)
    Y = np.abs(lat - y)
    return np.where((X == X.min()) & (Y == Y.min()))


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. This is similar to
    the approach used in the original U-Net paper.
    https://arxiv.org/abs/1505.04597
    MLX does not yet support transposed convolutions as of 0.6.0 so this is necessary.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = self.conv(upsample_nearest(x))
        return x


class Encoder(nn.Module):
    """
    A convolutional variational encoder.
    Maps the input to a normal distribution in latent space and sample a latent
    vector from that distribution.
    """

    def __init__(self, num_latent_dims, image_shape, max_num_filters):
        super().__init__()

        # number of filters in the convolutional layers
        num_filters_1 = max_num_filters // 4
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters

        # Output (BHWC):  B x 32 x 32 x num_filters_1
        self.conv1 = nn.Conv2d(image_shape[-1], num_filters_1, 3, stride=2, padding=1)
        # Output (BHWC):  B x 16 x 16 x num_filters_2
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, 3, stride=2, padding=1)
        # Output (BHWC):  B x 8 x 8 x num_filters_3
        self.conv3 = nn.Conv2d(num_filters_2, num_filters_3, 3, stride=2, padding=1)

        # Batch Normalization
        self.bn1 = nn.BatchNorm(num_filters_1)
        self.bn2 = nn.BatchNorm(num_filters_2)
        self.bn3 = nn.BatchNorm(num_filters_3)

        # Divide the spatial dimensions by 8 because of the 3 strided convolutions
        output_shape = [num_filters_3] + [
            dimension // 8 for dimension in image_shape[:-1]
        ]

        flattened_dim = math.prod(output_shape)

        # Linear mappings to mean and standard deviation
        self.proj_mu = nn.Linear(flattened_dim, num_latent_dims)
        self.proj_log_var = nn.Linear(flattened_dim, num_latent_dims)

    def __call__(self, x):
        x = nn.leaky_relu(self.bn1(self.conv1(x)))
        x = nn.leaky_relu(self.bn2(self.conv2(x)))
        x = nn.leaky_relu(self.bn3(self.conv3(x)))
        x = mx.flatten(x, 1)  # flatten all dimensions except batch

        mu = self.proj_mu(x)
        logvar = self.proj_log_var(x)
        # Ensure this is the std deviation, not variance
        sigma = mx.exp(logvar * 0.5)

        # Generate a tensor of random values from a normal distribution
        eps = mx.random.normal(sigma.shape)

        # Reparametrization trick to brackpropagate through sampling.
        z = eps * sigma + mu

        return z, mu, logvar


class Decoder(nn.Module):
    """
    A convolutional decoder
    """
    def __init__(self, num_latent_dims, image_shape, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        num_img_channels = image_shape[-1]
        self.max_num_filters = max_num_filters

        # decoder layers
        num_filters_1 = max_num_filters
        num_filters_2 = max_num_filters // 2
        num_filters_3 = max_num_filters // 4

        # divide the last two dimensions by 8 because of the 3 upsampling convolutions
        self.input_shape = [dimension // 8 for dimension in image_shape[:-1]] + [
            num_filters_1
        ]
        flattened_dim = math.prod(self.input_shape)

        # Output: flattened_dim
        self.lin1 = nn.Linear(num_latent_dims, flattened_dim)
        # Output (BHWC):  B x 16 x 16 x num_filters_2
        self.upconv1 = UpsamplingConv2d(
            num_filters_1, num_filters_2, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 32 x 32 x num_filters_1
        self.upconv2 = UpsamplingConv2d(
            num_filters_2, num_filters_3, 3, stride=1, padding=1
        )
        # Output (BHWC):  B x 64 x 64 x #img_channels
        self.upconv3 = UpsamplingConv2d(
            num_filters_3, num_img_channels, 3, stride=1, padding=1
        )

        # Batch Normalizations
        self.bn1 = nn.BatchNorm(num_filters_2)
        self.bn2 = nn.BatchNorm(num_filters_3)

    def __call__(self, z):
        x = self.lin1(z)

        # reshape to BHWC
        x = x.reshape(
            -1, self.input_shape[0], self.input_shape[1], self.max_num_filters
        )

        # approximate transposed convolutions with nearest neighbor upsampling
        x = nn.leaky_relu(self.bn1(self.upconv1(x)))
        x = nn.leaky_relu(self.bn2(self.upconv2(x)))
        # sigmoid to ensure pixel values are in [0,1]
        x = mx.sigmoid(self.upconv3(x))
        return x


class CVAE(nn.Module):
    """
    A convolutional variational autoencoder consisting of an encoder and a
    decoder.
    """

    def __init__(self, num_latent_dims, input_shape, max_num_filters):
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self.encoder = Encoder(num_latent_dims, input_shape, max_num_filters)
        self.decoder = Decoder(num_latent_dims, input_shape, max_num_filters)

    def __call__(self, x):
        # image to latent vector
        z, mu, logvar = self.encoder(x)
        # latent vector to image
        x = self.decode(z)
        return x, mu, logvar

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)



def rasterio_transform_to_gdal_transform(rio_transform):
    """
    Converts rasterio transform to gdal transform
    """
    return (
        rio_transform[2],
        rio_transform[0],
        rio_transform[1],
        rio_transform[5],
        rio_transform[3],
        rio_transform[4],
    )


def coord_to_pixel(x, y, transform, err=0.1):
    """From geospatial coordinate (x,y) to image pixel (row,col).
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to
    calculate the pixel location of a geospatial coordinate.
    x/y and transform must be in the same georeference.
    err is truncation error to return integer indices.
        if err=None, return pixel coords as floats.
    """
    if np.ndim(x) == 0:
        x, y = np.array([x]), np.array([y])
    col = (x - transform[0]) / transform[1]
    row = (y - transform[3]) / transform[5]
    if err:
        row = (row + err).astype(int)
        col = (col + err).astype(int)
    return row.item(), col.item()


def pixel_to_coord(row, col, transform):
    """From image pixel (i,j) to geospatial coordinate (x,y)."""
    if np.ndim(row) == 0:
        row, col = np.array([row]), np.array([col])
    x = transform[0] + col * transform[1]
    y = transform[3] + row * transform[5]
    return x.item(), y.item()


########################################################################################
############################ MLX OBJECT DETECTION UTILS ############################
########################################################################################



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
