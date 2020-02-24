import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import gdown
import dnnlib.tflib as tflib
from dnnlib.util import open_url
from utils.utils import split_to_batches_concurrent, create_concurrent_image_lists
from encoder.generator_model import Generator
from encoder.perceptual_model_concurrent import PerceptualModelConcurrent, load_images
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

# Pretrained models URL
url_styleGAN = 'https://drive.google.com/uc?export=download&id=1Ru1kpacSvmheTHP7evEGHEegXZjeTaoi'
url_resnet = 'https://drive.google.com/uc?id=1aT59NFy9-bNyXjDuZOTMl0qX0jmZc6Zb'
url_VGG_perceptual = 'https://drive.google.com/uc?export=download&id=1poMANPSNDHALZRuaqJGrl1EVOP1WNjLv'


def main():
    parser = argparse.ArgumentParser(description='Find latent representation of reference images using perceptual losses', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Output directories setting
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('guessed_images_dir', help='Directory for storing initially guessed images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')

    # General params
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)
    parser.add_argument('--use_resnet', default=True, help='Use pretrained ResNet for approximating dlatents', type=lambda x: (str(x).lower() == 'true'))

    # Perceptual model params
    parser.add_argument('--iterations', default=100, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--lr', default=0.02, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
    parser.add_argument('--decay_steps', default=10, help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=256, help='Size of images for the Resnet model', type=int)

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss', default=1.5, help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_mssim_loss', default=100, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_l1_penalty', default=1, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.', type=float)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--clipping_threshold', default=2.0, help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
    parser.add_argument('--face_mask', default=False, help='Generate a mask for predicting only the face area', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--use_grabcut', default=True, help='Use grabcut algorithm on the face mask to better segment the foreground', type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--scale_mask', default=1.5, help='Look over a wider section of foreground for grabcut', type=float)

    args, other_args = parser.parse_known_args()

    args.decay_steps *= 0.01 * args.iterations  # Calculate steps as a percent of total iterations

    # create reference images lists
    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = sorted(list(filter(os.path.isfile, ref_images)))
    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)
    ref_images_1, ref_images_2 = create_concurrent_image_lists(ref_images)

    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.guessed_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    if args.face_mask:
        os.makedirs(args.mask_dir, exist_ok=True)

    # Initialize generator
    tflib.init_tf()
    with open_url(url_styleGAN, cache_dir='cache') as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    generator = Generator(model=Gs_network,
                          batch_size=args.batch_size,
                          clipping_threshold=args.clipping_threshold,
                          tiled_dlatent=args.tile_dlatents,
                          model_res=args.model_res,
                          randomize_noise=args.randomize_noise)

    # Initialize perceptual model
    perc_model = None
    if args.use_lpips_loss > 1e-7:
        with open_url(url_VGG_perceptual, cache_dir='cache') as f:
            perc_model = pickle.load(f)
    perceptual_model = PerceptualModelConcurrent(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator)

    # Initialize ResNet model
    resnet_model = None
    if args.use_resnet:
        print("\nLoading ResNet Model:")
        resnet_model_fn = 'data/finetuned_resnet.h5'
        gdown.download(url_resnet, resnet_model_fn, quiet=True)
        resnet_model = load_model(resnet_model_fn)

    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    for images_batch_1, images_batch_2 in tqdm(split_to_batches_concurrent(ref_images_1, ref_images_2, args.batch_size),
                                               total=len(ref_images_1) // args.batch_size):
        names = [os.path.splitext(os.path.basename(i1))[0]+'_vs_'+os.path.splitext(os.path.basename(i2))[0]
                 for i1, i2 in zip(images_batch_1, images_batch_2)]
        perceptual_model.set_reference_images(images_batch_1, images_batch_2)

        # predict initial dlatents with ResNet model
        if resnet_model is not None:
            dlatents_1 = resnet_model.predict(preprocess_input(load_images(images_batch_1, image_size=args.resnet_image_size)))
            dlatents_2 = resnet_model.predict(preprocess_input(load_images(images_batch_2, image_size=args.resnet_image_size)))
            dlatents = 0.5 * (dlatents_1 + dlatents_2)
            generator.set_dlatents(dlatents)

        # Generate and save initially guessed images
        initial_dlatents = generator.get_dlatents()
        initial_images = generator.generate_images()
        for img_array, dlatent, img_name in zip(initial_images, initial_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.guessed_images_dir, f'{img_name}.png'), 'PNG')

        # Optimization process to find best latent vectors
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations)
        progress_bar = tqdm(op, leave=False, total=args.iterations)
        best_loss = None
        best_dlatent = None
        for loss_dict in progress_bar:
            progress_bar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v) for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                best_loss = loss_dict["loss"]
                best_dlatent = generator.get_dlatents()
            generator.stochastic_clip_dlatents()
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        # Generate images from found dlatents and save them
        generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)
        generator.reset_dlatents()

    # Concatenate and save dlalents vectors
    list_dlatents = sorted(os.listdir(args.dlatent_dir))
    final_w_vectors = np.array([np.load('latent_representations/' + dlatent) for dlatent in list_dlatents])
    np.save(os.path.join(args.dlatent_dir, 'output_vectors.npy'), final_w_vectors)


if __name__ == "__main__":
    main()