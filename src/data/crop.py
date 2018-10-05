from pathlib import Path
import numpy as np
import PIL

import cv2

import src.utils.utils as ut


def cropper(image):
    """
    poormans cropping routine: select region which only shows the targets
    """
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    image_rgb = cv2.fastNlMeansDenoising(image_rgb, None, 40, 29, 7)

    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 18, 100, 100)
    edged = cv2.Canny(gray, 20, 200, apertureSize=3)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    (_, cnts, _) = cv2.findContours(edged,
                                    cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt = None
    # loop over our contours
    app = []
    per = []
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        app.append(approx)
        per.append(peri)

    assert len(per) > 0
    assert len(app) > 0

    idx = np.argmax(np.array(per))
    screenCnt = np.array(app)[idx]

    rect = cv2.minAreaRect(screenCnt)
    cx, cy = rect[0]
    wi = ut.params.cropwidth
    ymin = np.max([int(cy-wi/2), 0])
    ymax = np.min([int(cy+wi/2), 900])
    xmin = np.max([int(cx-wi/2), 0])
    xmax = np.min([int(cx+wi/2), 900])
    cropped = image[ymin:ymax, xmin:xmax]
    return cropped


def crop_image(image, imagename=None):
    """cropping the image"""
    x, y, dx, dy = ut.params.croped_size
    wi = ut.params.cropwidth

    if(type(image) == PIL.Image.Image):
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    if image.size > 3*wi**2:
        cropped_1 = image[y:y+dy, x:x+dx]
        cropped_2 = cropper(cropped_1)
        if(imagename is not None):
            cv2.imwrite(imagename, cropped_2)
        return cropped_2
    else:
        return image


def crop():
    """run through the training directories"""
    for directory in [ut.dirs.train_dir,
                      ut.dirs.validation_dir,
                      ut.dirs.test_dir]:

        for feature in ['good', 'bad']:
            for file in Path(directory + '/' + feature).iterdir():
                if(file.name.endswith(('.jpeg'))):
                    imagename = directory + '/' + feature + '/' + file.name
                    image = cv2.imread(imagename)
                    crop_image(image, imagename)
