from PIL import Image
import os


def display_folder_content(folder, res=256):
    if folder[-1] != '/': folder += '/'
    for i, img_path in enumerate(sorted(os.listdir(folder))):
        img = Image.open(folder + img_path)
        w, h = img.size
        rescale_ratio = res / min(w, h)
        img = img.resize((int(rescale_ratio * w), int(rescale_ratio * h)), Image.LANCZOS)
        display(img, 'img %d: %s' % (i, img_path))
        print('\n')
