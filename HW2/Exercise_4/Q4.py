import cv2
import numpy as np
from HW2.Exercise_4.helper import Image, fix_images, make_lowpass_filter, cutoff, mix_near_and_far_frequency_domain, \
    mix_near_and_far_spatial_domain

# read images and create Image instance for them
source_near_image = cv2.imread(filename="../resources/Q4_01_near.jpg")
near_image = Image(source_near_image)
source_far_image = cv2.imread(filename="../resources/Q4_02_far.jpg")
far_image = Image(source_far_image)


# matches near and far images and save them
fixed_near_image, fixed_far_image = fix_images(near=near_image, far=far_image)
fixed_near_image.save(file_name="out/Q4_03_near.jpg")
fixed_far_image.save(file_name="out/Q4_04_far.jpg")


images_width, images_height = fixed_near_image.get_width(), fixed_near_image.get_height()


# fft_log images and save distribute version of them (0-255)
fft_log_near = fixed_near_image.get_fft_log_image()
fft_log_near.save_distributed(file_name="out/Q4_05_dft_near.jpg")

fft_log_far = fixed_far_image.get_fft_log_image()
fft_log_far.save_distributed(file_name="out/Q4_06_dft_far.jpg")


# create lowpass and highpass filters
lowpass_filter = make_lowpass_filter(width=images_width, height=images_height, D0=20)
distributed_lowpass_filter = lowpass_filter / lowpass_filter.max() * 255
cv2.imwrite(filename="out/Q4_08_lowpass_20.jpg", img=distributed_lowpass_filter)

highpass_filter = 1 - make_lowpass_filter(width=images_width, height=images_height, D0=20)
distributed_highpass_filter = highpass_filter / highpass_filter.max() * 255
cv2.imwrite(filename="out/Q4_07_highpass_20.jpg", img=distributed_highpass_filter)


# cutoff
cutoff_filter_for_lowpass = cutoff(width=images_width, height=images_height, D0=20)
lowpass_cutoff = np.multiply(cutoff_filter_for_lowpass, lowpass_filter)
distributed_lowpass_cutoff = lowpass_cutoff / lowpass_cutoff.max() * 255
cv2.imwrite(filename="out/Q4_10_lowpass_cutoff.jpg", img=distributed_lowpass_cutoff)

cutoff_filter_for_highpass = 1 - cutoff(width=images_width, height=images_height, D0=15)
highpass_cutoff = np.multiply(cutoff_filter_for_highpass, highpass_filter)
distributed_highpass_cutoff = highpass_cutoff / highpass_cutoff.max() * 255
cv2.imwrite(filename="out/Q4_09_highpass_cutoff.jpg", img=distributed_highpass_cutoff)


# near in frequency domain
near_highpassed_frequency = fixed_near_image.get_image_of_applied_cutoff_filter_in_frequency_domain(highpass_cutoff)
near_highpassed_frequency.save("out/Q4_11_highpassed.jpg")
# near in frequency doamin
near_highpassed_spatial = fixed_near_image.get_image_of_applied_cutoff_filter_in_spatial_domain(
    highpass_cutoff)
near_highpassed_spatial.save("out/Q4_11_hipassed_spatial_domain.jpg")

# far in frequency domain
far_lowpassed_frequency = fixed_far_image.get_image_of_applied_cutoff_filter_in_frequency_domain(lowpass_cutoff)
far_lowpassed_frequency.save("out/Q4_12_lowpassed.jpg")
# far in spatial domain
far_lowpassed_spatial = fixed_far_image.get_image_of_applied_cutoff_filter_in_spatial_domain(
    lowpass_cutoff)
far_lowpassed_spatial.save("out/Q4_12_lowpassed_spatial_domain.jpg")


near_highpassed_channels = fixed_near_image.apply_cutoff_filter_on_channels_in_frequency_domain(highpass_cutoff, "complex")
far_lowpassed_channels = fixed_far_image.apply_cutoff_filter_on_channels_in_frequency_domain(lowpass_cutoff, "complex")

mix_frequency, mix_channels = mix_near_and_far_frequency_domain(
    near_channels=near_highpassed_channels,
    far_channels=far_lowpassed_channels, alpha=1.5, beta=1)

cv2.imwrite(filename="out/Q4_13_hybrid_frequency.jpg", img=mix_frequency)


mix_spatial = mix_near_and_far_spatial_domain(mix_channels, images_width, images_height)
cv2.imwrite(filename="out/Q4_14_hybrid_near.jpg", img=mix_spatial)


far_resized_shape = (images_width // 10, images_height // 10)
far_resized = cv2.resize(mix_spatial, far_resized_shape, interpolation=cv2.INTER_AREA)
cv2.imwrite(filename="out/Q4_15_hybrid_far.jpg", img=far_resized)
