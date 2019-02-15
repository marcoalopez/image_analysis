# ============================================================================ #
#                                                                              #
#    Image analysis                                                            #
#    A Python code snippets for doing image analysis                           #
#                                                                              #
#    Copyright (c) 2019-present   Marco A. Lopez-Sanchez                       #
#                                                                              #
#    This Source Code Form is subject to the terms of the Mozilla Public       #
#    License, v. 2.0. If a copy of the MPL was not distributed with this       #
#    file, You can obtain one at http://mozilla.org/MPL/2.0/.                  #
#                                                                              #
#    Covered Software is provided under this License on an “AS IS” BASIS,      #
#    WITHOUT WARRANTY OF ANY KIND, either expressed, implied, or statutory,    #
#    including, without limitation, warranties that the Covered Software is    #
#    FREE OF DEFECTS, merchantable, fit for a particular purpose or            #
#    non-infringing. The entire risk as to the quality and performance         #
#    of the Covered Software is with You. Should any Covered Software prove    #
#    defective in any respect, You (not any Contributor) assume the cost of    #
#    any necessary servicing, repair, or correction. This disclaimer of        #
#    warranty constitutes an essential part of this License. No use of any     #
#    Covered Software is authorized under this License except under this       #
#    disclaimer.                                                               #
#                                                                              #
#    Version alpha                                                             #
#    For details see: https://github.com/marcoalopez/image_analysis            #
#                                                                              #
#    Requirements:                                                             #
#        Python version 3.5 or higher                                          #
#        Numpy version 1.11 or higher                                          #
#        Matplotlib version 2.0 or higher                                      #
#        Pillow version                                                        #
#        dcraw installed in the system (for the raw2tiff function)             #
#                                                                              #
# ============================================================================ #


# Import neccesary libraries
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def raw2tiff(path='auto',
             dcraw_arg='C:/Users/Marco/Documents/dcraw/dcraw64.exe -v -w -H 0 -o 0 -q 3 -T',
             raw_format='.NEF'):
    """ Automate the conversion from RAW to other image format using the
    script dcraw by Dave Coffin's.

    Parameters
    ----------
    path : string
        the file path where the images to be converted are. If 'auto',
        the default, the function will ask you for the folder location
        through a file selection dialog.

    dcraw_arg : string
        the file path where the dcraw executable is and the dcraw
        arguments. For details on the dcraw arguments call dcraw
        in the console or go to:
        https://www.cybercom.net/~dcoffin/dcraw/dcraw.1.html

    raw_format : string
        the format of the raw images. Default: '.NEF'

    Returns
    -------
    None

    Requirements
    ------------
    dcraw installed in the system
    """

    if path == 'auto':
        path = get_path()

    for filename in os.listdir(path):
        if filename.endswith(raw_format):
            print('converting {}' .format(filename))
            subprocess.run(dcraw_arg + ' ' + path + filename, shell=False)

    print(' ')
    print('Done!')

    return None


def RGB2gray(input_path='auto',
             output_path='auto',
             input_format='.tiff',
             output_format='.tif'):
    """ Automatically convert RGB images to 8-bit grayscale images
    cointained in a specific folder.

    Parameters
    ----------
    input_path : string
        the file path where the images to be converted are stored.
        If 'auto', the default, the function will ask you for the
        folder location through a file selection dialog.

    output_path : string
        the file path where the grayscale images will be saved.
        If 'auto', the default, the function will ask you for the
        folder location through a file selection dialog.

    input_format : string
        the format of the input images. Default: '.tiff'

    output_format : string
        the format of the input images. Default: '.tif'

    Returns
    -------
    None
    """

    if input_path == 'auto':
        input_path = get_path()

    if output_path == 'auto':
        output_path = get_path()

    for filename in os.listdir(input_path):
        if filename.endswith(input_format):
            print('converting {} and saving' .format(filename))
            img = Image.open(input_path + filename).convert('L')
            fn, ext = os.path.splitext(filename)  # separate name and ext
            img.save(output_path + fn + output_format)

    print(' ')
    print('Done!')

    return None


def denoising_img_avg(save_as='denoise_img.tif',
                      path='auto',
                      file_type='.tif',
                      robust=True,
                      noise_floor=False):
    """ Noise reduction by image averaging. Images should be aligned.
    By noise we refer here to stochastic variations produced in CCD or
    CMOS sensors.

    Parameters
    ----------
    save_as : string
        the name of the generated image, the format to use is determined from
        the filename extension

    path : string
        the path to the folder containing the images to average. If 'auto',
        the default, the function will ask you for the folder location
        through a file selection dialog.

    file_type : string
        the image format to read

    robust : bool, default True
        if True the averaging method use the median. If false the mean.

    noise_floor : bool, default False
        if True, the noise floor is calculated.

    Returns
    -------
    a numpy array with denoised image (and the std values of each pixel
    if noise_floor=True). The latter is useful is you want to locate pixels
    that yield large errors.

    Examples
    --------
    >>> denoising_img_avg(path='C:/Users/name/Documents/my_images/')
    >>> denoising_img_avg(save_as='new_image.tif', path='C:/Users/name/Documents/my_images/')

    # For estimate and visualize the noise floor of the image do
    >>> denoise_img, std_px_vals = denoising_img_avg(noise_floor=True)
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(std_px_vals, cmap='cividis')
    >>> fig.colorbar(im, ax=ax)
    """

    if path == 'auto':
        path = get_path()

    # open and stack all images
    print(' ')
    print('Stacking images...')
    count = 0
    for filename in os.listdir(path):
        if filename.endswith(file_type):
            img = np.array(Image.open(path + filename).convert('L'))
            if count == 0:
                img_stack = img
            else:
                img_stack = np.dstack((img_stack, img))
            count += 1

    # denoise by averaging
    print(' ')
    print('Denoising image...')
    if robust is True:
        denoise_img = np.median(img_stack, axis=2)
    else:
        denoise_img = np.mean(img_stack, axis=2)

    # convert from float to integer
    denoise_img = np.uint8(denoise_img)

    # Estimate the noise floor if proceed
    if noise_floor is True:
        # TODO: ask whether the the noise floor is for a ROI
        crop = input("Do you want to define a region of interest (ROI) to estimate the noise floor? (type 'y' or 'n'): ")
        if crop == 'y':
            x, y, xp, yp = input("Define the coordinates (x, y, x', y') (e.g. (1014, 192, 4692, 4644): ")
            px_std_vals = np.std(img_stack[x - 1:y, xp - 1:yp, :], axis=2)

        print(' ')
        print('Estimating the noise floor...')
        px_std_vals = np.std(img_stack, axis=2)
        mean_std = np.mean(px_std_vals)
        print(' ')
        print('Noise floor:')
        print('Mean of SD values =', round(mean_std, 2))
        print('max, min =', round(np.max(px_std_vals), 2),
              ',', round(np.min(px_std_vals)))

    # plot the image using matplotlib
    fig, ax = plt.subplots()
    im = ax.imshow(denoise_img, cmap='bone')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    # save the denoised image in the same path
    img = Image.fromarray(denoise_img)
    img.save(path + save_as)

    if noise_floor is True:
        return denoise_img, px_std_vals
    else:
        return denoise_img


def img_autocorrelation(image, plot=True):
    """ Compute the autocorrelation of a image via Fourier transform

    Parameters
    ----------
    image : ndarray
        Reference 8-bit gray image stored as ndarray

    plot : bool
        If true (default), make the plots

    Returns
    -------
    TODO

    Call functions
    --------------
    auto_corr_plot()
    TODO

    Example
    -------
    >>> image = np.array(Image.open('image.tif').convert('L'))
    >>> img_autocorrelation(image)
    """

    # check type and proceed TODO
    if type(image) is not np.ndarray:
        raise ValueError('Error: image must be entered as numpy.ndarray')

    # generate an image of square size (nm * nm)
    nm = np.min(np.shape(image))
    data = image[:nm, :nm]

    # Estimate the mean gray value
    mean_gray = np.mean(data)

    # Pad the array with zeros. This allows to obtain a smoother spectrum
    # whenplotting the Fourier transform. It adds zero to the edges (last
    # column and row) of the array/image/matrix. TODO-> Use np.pad instead
    data_padded = np.zeros([nm + 1, nm + 1])
    data_padded[:nm, :nm] = data

    # Compute the 2D discrete Fourier transform of the image/matrix
    fft_image = np.fft.fft2(data_padded)
    abs_fft = np.abs(fft_image)**2
    ifft_image = np.fft.ifft2(abs_fft)

    # Compute autocorrelation
    autocorrelation = np.abs(np.fft.fftshift(ifft_image / np.nanmax(ifft_image)))
    c_inf = mean_gray**2 / np.mean(data_padded**2)  # TODO: this is why others pad with the mean gray!

    # extract radial profiles
    y, x = np.indices(autocorrelation.shape)
    centre = np.array([(x.max() - x.min()) / 2,
                       (y.max() - y.min()) / 2])

    # Note: The autocorrelation function is the inverse Fourier Transform of the power spectrum

    # plot the 2D power spectrum (if apply)
    if plot is True:
        spectrum_plot(autocorrelation, profile)  # TODO

    pass


def spectrum_plot():

    pass


def make_video(path='auto',
               ffmpeg_path='C:/Users/Marco/Documents/ffmpeg/bin/',
               ffmpeg_arg='ffmpeg -r 1 -f image2 -s 1280x720 -i DSC_%04d.tif -vcodec libx264 -crf 25 movie.mp4'):
    """ Make a video from a sequence of images using the the ffmpeg converter

    Parameters
    ----------
    path : string
        the path whre the sequence of images are

    ffmpeg_path : string
        the path where the ffmpeg executable are

    ffmpeg_arg : string
        the ffmpeg arguments. For example:
        http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/

    Returns
    -------
    None

    Requirements
    ------------
    ffmpeg installed in the system
    """

    if path == 'auto':
        path = get_path()

    os.chdir(path)
    subprocess.run(ffmpeg_path + ffmpeg_arg, shell=False)

    print(' ')
    print('Done!')

    return None


def get_path():
    """ Get the folder path through a file selection dialog.
    It uses tkinter"""

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        file_path = filedialog.askdirectory()
    except ImportError:
        print('The script requires Python 3.5 or higher')

    return file_path + '/'
