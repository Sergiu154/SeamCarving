import sys
import cv2 as cv
import numpy as np
import copy

from cod.parameters import *
from cod.select_path import *

import pdb


def compute_energy(img, roi_mask=None):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param
    img: imaginea initiala
    roi_mask: o masca ce dimensiunea imaginii curente ce contine in zona selectata niste weights negative
              ce vor fi adaugate la map-ul de energie E
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea pentru fiecare pixel al imaginii
    E = np.zeros((img.shape[0], img.shape[1]))

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobelx = cv.Sobel(img_gray, cv.CV_64F, 1, 0)

    sobely = cv.Sobel(img_gray, cv.CV_64F, 0, 1)

    E = np.abs(sobelx) + np.abs(sobely)

    # daca suntem la punctul cu eliminare obiectelor
    # atunci aduna o masca cu valori negative in regiunea de interes(de eliminat)
    # pentru a forta algoritmul de programare dinamica sa treaca prin regiunea selectata de fiecare data
    # pana ce se elimina intreaga regiune

    if roi_mask is not None:
        E += roi_mask

    return E


def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:, :, 0] = E.copy()
    new_image_E[:, :, 1] = E.copy()
    new_image_E[:, :, 2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    """
    elimina drumul vertical din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    # adaptam aceasta functie pentru a redimensiuna si roi_mask

    # daca este imaginea de input
    if len(img.shape) == 3:
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    elif len(img.shape) == 2:
        # daca este roi_mask atunci este 2d
        updated_img = np.zeros((img.shape[0], img.shape[1] - 1), np.float64)

    for i in range(img.shape[0]):
        col = path[i][1]
        # copiem partea din stanga
        updated_img[i, :col] = img[i, :col].copy()
        # copiem partea din dreapta
        updated_img[i, col:] = img[i, (col + 1):].copy()
    return updated_img


def decrease_width(params: Parameters, num_pixels, roi=None):
    img = params.image.copy()  # copiaza imaginea originala

    roi_mask = None
    # daca trebuie sa eliminam un obiect
    if roi:
        # creeaza o masca plina cu zero-uri
        # dar care in regiunea selectata are weight-uri negative
        roi_mask = np.zeros((params.image.shape[0], params.image.shape[1]))
        roi_mask[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])] = -10 ** 10

    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img, roi_mask)

        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)

        img = delete_path(img, path)
        # daca sunt la eliminarea obiectului atunci elimina si coloana stearsa din roi_mask
        if roi_mask is not None:
            roi_mask = delete_path(roi_mask, path)

    cv.destroyAllWindows()
    return img


def decrease_height(params: Parameters, num_pixels, roi=None):
    # vom roti imaginea cu 90 de grade clockwise si aplicam alg de eliminare pe latime
    params.image = cv.rotate(params.image.copy(), cv.ROTATE_90_CLOCKWISE)
    return decrease_width(params, num_pixels, roi)


def delete_object(params: Parameters, x0, y0, w, h):
    if w < h:
        # functia roi returneaza (y0,x0,w,h)
        return decrease_width(params, w, roi=(y0, x0, w, h))
    else:
        # rotind imaginea 90 grade clockwise avem
        new_x0 = y0  # r[1]  noua coord.a linie = coordonata coloanei
        new_y0 = params.image.shape[0] - x0 - h  # r[0]
        # noua coord. a coloanei devine (latimea(i.e fosta lungime) -

        return cv.rotate(decrease_height(params, h, roi=(new_y0, new_x0, h, w)),
                         cv.ROTATE_90_COUNTERCLOCKWISE)


def amplify_content(params):
    H, W = params.image.shape[:2]
    # noile dimensiuni dupa amplificare
    new_H = int(H * params.factor_amplification)
    new_W = int(W * params.factor_amplification)

    # redimensiuneaza imaginea initiala
    params.image = cv.resize(params.image, (new_W, new_H))

    # calculeaza numarul de pixeli ce trebuie eliminati pe orizontala si verticala
    params.num_pixel_height = new_H - H
    params.num_pixels_width = new_W - W
    # elimina pe latime si roteste rezultatul 90 de grade
    # pentru a-l da alg de eliminare pe inaltime
    params.image = decrease_width(params, params.num_pixels_width)
    final_image = decrease_height(params, params.num_pixel_height)
    # roteste rezultatul final la pozitia initiala
    params.image = cv.rotate(params.image, cv.ROTATE_90_COUNTERCLOCKWISE)

    return cv.rotate(final_image, cv.ROTATE_90_COUNTERCLOCKWISE)


def resize_image(params: Parameters):
    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':

        # roteste imaginea rezultat la loc
        res = cv.rotate(decrease_height(params, params.num_pixel_height), cv.ROTATE_90_COUNTERCLOCKWISE)
        # rotate back the originla image to be correctly display in plot
        params.image = cv.rotate(params.image, cv.ROTATE_90_COUNTERCLOCKWISE)
        return res

    elif params.resize_option == 'amplificaContinut':
        return amplify_content(params)

    elif params.resize_option == 'eliminaObiect':
        r = cv.selectROI('Fereastra', params.image.astype(np.uint8))

        rez = delete_object(params, r[1], r[0], r[2], r[3])
        # daca inaltimea ferestre e mai mica de latimea atunci roteste
        # imaginea la loc la final
        if r[3] < r[2]:
            params.image = cv.rotate(params.image, cv.ROTATE_90_COUNTERCLOCKWISE)
        return rez

    else:
        print('The option is not valid!')
        sys.exit(-1)
