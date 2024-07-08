from PIL import Image
from numpy import average, dot, linalg

# normalization
def get_thum(image, size=(1920, 1080), greyscale=False):
    image = image.resize(size, Image.Resampling.LANCZOS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image_cp1, image_cp2):
    image_cp1 = get_thum(image_cp1)
    image_cp2 = get_thum(image_cp2)
    images = [image_cp1, image_cp2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
        
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
