import numpy as np
import numpy.linalg as la
from numpy.random import default_rng

import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.pyplot import figure, imshow, axis

from mpl_toolkits.axes_grid1 import ImageGrid

global N 
N = 92

global M 
M = 112

global NM 
NM = N * M

global threshold
threshold = 0

faces_dir = "./resources/att_faces"
rng = default_rng()

def set_threshold(theta: float):
    global threshold
    threshold = theta
    print(threshold)

def get_path_to_img(sid: int, img_id: int):
    return faces_dir + "/s" + str(sid) + "/" + str(img_id) + ".pgm"

# Returns a 1D array with the image data
def get_img(path: str):
    image = imread(path)
    return np.array(image, dtype = float).flatten()

def render_images(images: np.ndarray):
    fig = figure(figsize=(M, N))    
    nr_imgs = images.shape[1]
    
    rows = nr_imgs // 10 if nr_imgs % 10 == 0 else nr_imgs // 10 + 1
    
    grid = ImageGrid(fig, 111, 
                nrows_ncols=(rows, 10),
                axes_pad=0.1,
                )
    for ax, im in zip(grid, images.T):
        ax.imshow(im.reshape(M, N), cmap = 'Greys_r')
        ax.axis('off')

def render_image(image: np.ndarray):
    imshow(image.reshape(M, N), cmap = 'Greys_r')
    plt.axis('off')

# Returns a matrix with the given number of faces and the ids used: (face_matrix, training_set_ids)
def load_training_set(nr_of_subjects: int):
    training_set_ids = rng.choice(range(1, 41), size = nr_of_subjects, replace = False)
    face_matrix = np.zeros(shape = (NM, nr_of_subjects * 8), dtype = float)
    k = 0
    for id in training_set_ids:
        for j in range(1, 9):
            path_to_img = get_path_to_img(id, j)
            image_vector = get_img(path_to_img)
            face_matrix[:, k] = image_vector
            k += 1
    return face_matrix, training_set_ids

def get_top_N_eigenfaces(N: int, evalues: np.ndarray, efaces: np.ndarray):
    top_indices = list(reversed(evalues.argsort()))
    top_N_eigenfaces = np.zeros((NM, N), dtype = float)
     
    for i, j in enumerate(top_indices[:N]):
        top_N_eigenfaces[:, i] = efaces[:, j]
    
    render_images(top_N_eigenfaces)
    
    return top_N_eigenfaces


def get_weights_vector(basis_eigenfaces: np.ndarray, image: np.ndarray):
    return basis_eigenfaces.T @ image

# Returns a list with the weights of every training image of a subject 
def get_subject_weights(basis_eigenfaces: np.ndarray, mean_face: np.ndarray, subject_id: int):
    weights = np.zeros((basis_eigenfaces.shape[1], 8), dtype = float)
    for i in range(1, 9):
        image_vector = get_img(get_path_to_img(subject_id, i))
        image_vector -= mean_face
        weights[:, i - 1] = get_weights_vector(basis_eigenfaces, image_vector)
    
    return weights

def distance_to_face_class(face_classes: dict, kth_class: int, weights_vector: np.ndarray):
    return la.norm(weights_vector - face_classes[kth_class])

def set_threshold(theta: float):
    global threshold
    threshold = theta
    print(threshold)

# Returns (class, distance)
def get_closest_face_class(face_classes: dict, weights_vector: np.ndarray):
    ret_class = 0
    min_dist = float('inf')
    
    for k in face_classes:
        dist = distance_to_face_class(face_classes, k, weights_vector)
        if (dist < min_dist):
            min_dist = dist
            ret_class = k
            
    return ret_class, min_dist

# Tries to recognize the image with the given weights.
def try_to_recognize_image(face_classes: dict, weights_vector: np.ndarray, subject_id: int, img_id: int):
    k, dist_to_a_face_class = get_closest_face_class(face_classes, weights_vector)
    if dist_to_a_face_class < threshold:
        print("Image {}-{} is subject {}!".format(subject_id, img_id, k))
        
        if subject_id != k:
            return "FP"
        else:
            return "P"
            
    elif dist_to_a_face_class >= threshold:
        print("Image {}-{} is an unknown subject!".format(subject_id, img_id))
        return "UNK"

def recognition_test(face_classes: dict, training_set_ids: np.ndarray,
                        basis_eigenfaces: np.ndarray, mean_face: np.ndarray):
    positives = 0
    false_positives = 0
    unknown_faces = 0

    # Recognition loop
    for sub in training_set_ids:
        for img in range(9, 11):
            path_to_img = get_path_to_img(sub, img)
            image_vector = get_img(path_to_img)
            image_vector -= mean_face

            weights_vector = get_weights_vector(basis_eigenfaces, image_vector)

            res = try_to_recognize_image(face_classes, weights_vector, sub, img)

            if res == "P":
                positives += 1
            elif res == "FP":
                false_positives += 1
            elif res == "UNK":
                unknown_faces += 1
    return positives, false_positives, unknown_faces
