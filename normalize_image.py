# normalize_image.py
import histomicstk as htk
import skimage.io

def normalize_image(input_image, ref_image_file):
    # Load reference image for normalization
    im_reference = skimage.io.imread(ref_image_file)[:, :, :3]

    # Get mean and stddev of the reference image in lab space
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(im_reference)

    # Perform Reinhard color normalization
    normalized_image = htk.preprocessing.color_normalization.reinhard(input_image, mean_ref, std_ref)

    return normalized_image
