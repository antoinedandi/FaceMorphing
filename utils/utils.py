import os
import matplotlib.pyplot as plt
from PIL import Image
import dnnlib.tflib as tflib


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_to_batches_concurrent(l1, l2, n):
    assert len(l1) == len(l2)
    for i in range(0, len(l1), n):
        yield l1[i:i + n], l2[i:i + n]


def create_concurrent_image_lists(l):
    l1 = []
    l2 = []
    for i in range(len(l)):
        for j in range(i, len(l)):
            l1.append(l[i])
            l2.append(l[j])
    assert len(l1) == len(l2)
    return l1, l2


def display_folder_content(folder, res=256):
    if folder[-1] != '/': folder += '/'
    for i, img_path in enumerate(sorted(os.listdir(folder))):
        img = Image.open(folder + img_path)
        w, h = img.size
        rescale_ratio = res / min(w, h)
        img = img.resize((int(rescale_ratio * w), int(rescale_ratio * h)), Image.LANCZOS)
        display(img, 'img %d: %s' % (i, img_path))
        print('\n')


def display_encoding_results(original_imgs, guessed_imgs, reconstructed_imgs, res=256, fs=12):

    if original_imgs[-1] != '/': original_imgs += '/'
    if guessed_imgs[-1] != '/': guessed_imgs += '/'
    if reconstructed_imgs[-1] != '/': reconstructed_imgs += '/'

    imgs1 = sorted([f for f in os.listdir(original_imgs) if '.png' in f])
    imgs2 = sorted([f for f in os.listdir(guessed_imgs) if '.png' in f])
    imgs3 = sorted([f for f in os.listdir(reconstructed_imgs) if '.png' in f])

    for i in range(len(imgs1)):
        img1 = Image.open(original_imgs + imgs1[i]).resize((res, res))
        img2 = Image.open(guessed_imgs + imgs2[i]).resize((res, res))
        img3 = Image.open(reconstructed_imgs + imgs3[i]).resize((res, res))

        f, axarr = plt.subplots(1, 3, figsize=(fs, fs))
        axarr[0].imshow(img1)
        axarr[0].title.set_text('Original img %d' % i)
        axarr[1].imshow(img2)
        axarr[1].title.set_text('Initial guess img %d' % i)
        axarr[2].imshow(img3)
        axarr[2].title.set_text('Reconstructed img %d' % i)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()
        print("")


def generate_faces_from_latent(generator, latent_vector):
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
                            minibatch_size=1)
    batch_size = latent_vector.shape[0]
    return generator.components.synthesis.run(latent_vector.reshape((batch_size, 18, 512)),
                                              randomize_noise=False,
                                              **synthesis_kwargs)


def perform_face_morphing(w_vectors, generator, original_imgs, reconstructed_imgs, res=1024, fs=40):

    # Get aligned images and generated images
    imgs1 = sorted([f for f in os.listdir(original_imgs) if '.png' in f])
    imgs2 = sorted([f for f in os.listdir(reconstructed_imgs) if '.png' in f])

    n_images = len(imgs1)
    n_rows = n_images + 1
    n_cols = n_images + 2

    f, axes = plt.subplots(n_rows, n_cols, figsize=(fs, fs))
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    # Left Columns
    for i in range(n_images):
        img1 = Image.open(original_imgs + imgs1[i]).resize((res, res))
        img2 = Image.open(reconstructed_imgs + imgs2[i]).resize((res, res))
        axes[i + 1, 0].imshow(img1)
        axes[i + 1, 0].axis('off')
        axes[i + 1, 1].imshow(img2)
        axes[i + 1, 1].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')
            axes[i, 1].set_title('Reconstructed')

    # Top Row
    for j in range(n_images):
        img = Image.open(reconstructed_imgs + imgs2[j]).resize((res, res))
        axes[0, j + 2].imshow(img)
        axes[0, j + 2].axis('off')

    # Morphed Faces
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                axes[i + 1, j + 2].axis('off')
            else:
                avg_w = 0.5 * (w_vectors[i] + w_vectors[j]).reshape((1, 18, -1))
                img = generate_faces_from_latent(generator, avg_w)[0]
                axes[i + 1, j + 2].imshow(img)
                axes[i + 1, j + 2].axis('off')


def display_results_concurrent_optimization(original_imgs, reconstructed_imgs, res=1024, fs=40):
    
    # Get aligned images and generated images
    imgs1 = sorted([f for f in os.listdir(original_imgs) if '.png' in f])
    imgs2 = sorted([f for f in os.listdir(reconstructed_imgs) if '.png' in f])
    
    # Utility function to easily access morphed faces from the folder
    def get_morphed_image(i1, i2):
        fname = imgs1[i1] + '_vs_' + imgs1[i2]
        reconstructed_fnames = [os.path.splitext(os.path.basename(x))[0] for x in imgs2]
        assert fname in reconstructed_fnames
        return imgs2[reconstructed_fnames.index(fname)]

    n_images = len(imgs1)
    n_rows = n_images + 1
    n_cols = n_images + 2

    f, axes = plt.subplots(n_rows, n_cols, figsize=(fs, fs))
    plt.subplots_adjust(wspace=0, hspace=0)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')

    # Left Columns
    for i in range(n_images):
        img1 = Image.open(original_imgs + imgs1[i]).resize((res, res))
        img2 = Image.open(reconstructed_imgs + get_morphed_image(i, i)).resize((res, res))
        axes[i + 1, 0].imshow(img1)
        axes[i + 1, 0].axis('off')
        axes[i + 1, 1].imshow(img2)
        axes[i + 1, 1].axis('off')
        if i == 0:
            axes[i, 0].set_title('Original')
            axes[i, 1].set_title('Reconstructed')

    # Top Row
    for j in range(n_images):
        img = Image.open(reconstructed_imgs + get_morphed_image(j, j)).resize((res, res))
        axes[0, j + 2].imshow(img)
        axes[0, j + 2].axis('off')

    # Morphed Faces
    for i in range(n_images):
        for j in range(n_images):
            if i == j:
                axes[i + 1, j + 2].axis('off')
            else:
                img = Image.open(reconstructed_imgs + get_morphed_image(i, j)).resize((res, res))
                axes[i + 1, j + 2].imshow(img)
                axes[i + 1, j + 2].axis('off')
