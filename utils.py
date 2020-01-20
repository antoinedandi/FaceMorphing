import os
import matplotlib.pyplot as plt
from PIL import Image
import dnnlib.tflib as tflib


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

