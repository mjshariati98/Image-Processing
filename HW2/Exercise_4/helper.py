import math
import cv2
import numpy as np


class Image:

    def __init__(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    def save(self, file_name):
        cv2.imwrite(filename=file_name, img=self.image)

    def save_distributed(self, file_name):
        distributed_image = self.distribute_to_0_255()
        cv2.imwrite(filename=file_name, img=distributed_image)

    def get_shape(self):
        return self.image.shape

    def get_height(self):
        return self.image.shape[0]

    def get_width(self):
        return self.image.shape[1]

    def get_B_channel(self):
        return self.image[:, :, 0]

    def get_G_channel(self):
        return self.image[:, :, 1]

    def get_R_channel(self):
        return self.image[:, :, 2]

    def get_channels_list(self):
        return [self.get_B_channel(), self.get_G_channel(), self.get_R_channel()]

    def distribute_to_0_255(self):
        distributed_image = self.image / self.image.max() * 255
        uint8_image = np.asarray(distributed_image, dtype='uint8')

        return uint8_image

    def mix_channels(self, channels_list):
        """
        get a channel list that contains 3 channels (B, G, R in order). build a zero matrix of self image shape.
        then put every channel on new image channels (BGR_image) and return it.

        :param channels_list: list of channels to mix in order: B, G, R
        :return: new 'Image instance' with RGB format
        """
        BGR_image = np.zeros(self.get_shape())

        BGR_image[:, :, 0] = channels_list[0]
        BGR_image[:, :, 1] = channels_list[1]
        BGR_image[:, :, 2] = channels_list[2]

        new_image = Image(BGR_image)
        return new_image

    def get_shifted_fft_channels(self):
        """
        do this steps on every channel and put every result channel on a list and return it:
        1. fft.fft2
        2. fft.fftshift

        :return: a list that contain shifted fft of every channel of image.
        """
        shifted_fft_channels_list = []

        for channel in self.get_channels_list():
            fft_channel = np.fft.fft2(channel)
            shifted_fft_channel = np.fft.fftshift(fft_channel)

            shifted_fft_channels_list.append(shifted_fft_channel)

        return shifted_fft_channels_list

    def get_fft_log_channels(self):
        """
        use 'get_shifted_fft_channels' function and do these steps on result channels of 'get_shifted_fft_channels'
        function:
        1. np.abs on their channels
        2. np. log on their channels

        :return: a list contains log amplitude of every channel
        """
        log_amplitude_channels_list = []

        for channel in self.get_shifted_fft_channels():
            amplitude_channel = np.abs(channel)
            log_amplitude_channel = np.log(amplitude_channel)

            log_amplitude_channels_list.append(log_amplitude_channel)

        return log_amplitude_channels_list

    def get_fft_log_image(self):
        """
        get 'get_fft_log_channels' function result (log amplitude of every channels) and mix these channels and
        make a RGB image with function 'mix_channels'

        :return:
        """
        fft_log_channels = self.get_fft_log_channels()
        fft_log_image = self.mix_channels(fft_log_channels)

        return fft_log_image

    def apply_cutoff_filter_on_channels_in_frequency_domain(self, cutoff_filter, real_complex="complex"):
        """
        get 'get_shifted_fft_channels' result and multiply(element-wise) every channel with cutoff filter and return
        result in complex or real values.
        :param cutoff_filter: cutoff filter
        :param real_complex: a string: "complex" -> return channels with complex values, "real" -> return channels with
        real values.

        :return: a channel list that contains every channel after apply cutoff filter in frequency doamin
        """
        applied_channels_list = []

        shifted_fft_channels = self.get_shifted_fft_channels()
        for channel in shifted_fft_channels:
            applied_channel = np.multiply(channel, cutoff_filter)
            if real_complex == "real":
                real_applied = np.real(applied_channel)
                applied_channels_list.append(real_applied)
            else:
                applied_channels_list.append(applied_channel)

        return applied_channels_list

    def apply_cutoff_filter_on_channels_in_spatial_domain(self, cutoff_filter):
        """
        get 'get_shifted_fft_channels' result and multiply(element-wise) every channel with cutoff filter and
        then revese shift-fft and then reverse fft-fft2 and then result every channel in real values.

        :param cutoff_filter: cutoff filter
        :return: a channel list that contains every channel after apply cutoff filter and reverse to spatial domain
        """
        applied_channels_list = []

        shifted_fft_channels = self.get_shifted_fft_channels()
        for channel in shifted_fft_channels:
            applied_channel = np.multiply(channel, cutoff_filter)
            reverse_shifted_applied = np.fft.ifftshift(applied_channel)
            reverse_fft_applied = np.fft.ifft2(reverse_shifted_applied)
            real_applied = np.real(reverse_fft_applied)
            applied_channels_list.append(real_applied)

        return applied_channels_list

    def get_image_of_applied_cutoff_filter_in_frequency_domain(self, cutoff_filter):
        applied_channels = self.apply_cutoff_filter_on_channels_in_frequency_domain(cutoff_filter, 'real')
        mix_channels_image = self.mix_channels(applied_channels)

        return mix_channels_image

    def get_image_of_applied_cutoff_filter_in_spatial_domain(self, cutoff_filter):
        applied_channels = self.apply_cutoff_filter_on_channels_in_spatial_domain(cutoff_filter)
        mix_channels_image = self.mix_channels(applied_channels)

        return mix_channels_image


def fix_images(near, far):
    """
    get 2 instance of near and far and match them

    :param near: near image
    :param far: far image
    :return: (near, far) that are instances of Image and are fixed versions of near and far images
    """
    fixed_near = near.get_image()[:-10, 570:1327, :]
    near.set_image(fixed_near)

    fixed_far = far.get_image()[230:-46, :, :]
    far.set_image(fixed_far)

    return near, far


def gaussian_lowpass(x, y, D0):
    """
    gaussian lowpass function that explain in slides for fourier.

    :param x: the x-coordinate
    :param y: the y-coordinate
    :param D0: the D0
    :return:
    """
    D = math.sqrt(x ** 2 + y ** 2)
    return math.exp(-(D ** 2) / (2 * (D0 ** 2)))


def make_lowpass_filter(width, height, D0):
    """
    build filter(matrix) of gauusian lowpass funcion.

    :param width: width of filter to make
    :param height: height of filter to make
    :param D0: the D0
    :return: filter
    """
    filter = np.zeros((height, width))

    for i in range(-int(height / 2), int(height / 2)):
        for j in range(-int(width / 2), (width // 2)):
            filter[i + int(height / 2), j + (width // 2)] = gaussian_lowpass(j, i, D0)

    return filter


def cutoff(width, height, D0):
    """
    build a cutoff filter

    :param width: width of cutoff filter
    :param height: height of cutoff filter
    :param D0: the D0
    :return: filter
    """
    filter = np.zeros((height, width))
    for i in range(-int(height / 2), int(height / 2)):
        for j in range(-int(width / 2), (width // 2)):
            d = math.sqrt(i ** 2 + j ** 2)
            if d < D0:
                filter[i + int(height / 2), j + (width // 2)] = 1
    return filter


def mix_near_and_far_frequency_domain(near_channels, far_channels, alpha=1, beta=1):
    """
    get channels of applying highpassed-cutoff on near and lowpassed-cutoff on far images and sum them with factors
    alpha nad beta and return a tuple consist of these channels and mix image(build from mix channels)

    :param near_channels:
    :param far_channels:
    :param alpha: factor for near image
    :param beta: factor for far image
    :return: mix image and its channels in frequency domain
    """
    mix_near_and_far_channels = []

    mixed_near_far_B_channel = ((alpha * near_channels[0]) + (beta * far_channels[0])) / (
            alpha + beta)
    mixed_near_far_G_channel = ((alpha * near_channels[1]) + (beta * far_channels[1])) / (
            alpha + beta)
    mixed_near_far_R_channel = ((alpha * near_channels[2]) + (beta * far_channels[2])) / (
            alpha + beta)

    mix_near_and_far_channels.append(mixed_near_far_B_channel)
    mix_near_and_far_channels.append(mixed_near_far_G_channel)
    mix_near_and_far_channels.append(mixed_near_far_R_channel)

    width = mixed_near_far_B_channel.shape[1]
    height = mixed_near_far_B_channel.shape[0]
    mixed_near_far = np.zeros((height, width, 3))
    mixed_near_far[:, :, 0] = np.real(mixed_near_far_B_channel)
    mixed_near_far[:, :, 1] = np.real(mixed_near_far_G_channel)
    mixed_near_far[:, :, 2] = np.real(mixed_near_far_R_channel)

    return mixed_near_far, mix_near_and_far_channels


def mix_near_and_far_spatial_domain(mix_channels, width, height):
    """
    get 'mix_near_and_far_frequency_domain' function result (mixed near and far channels in frequency domain) and
    do these steps to reverse fourier transform:
    1. ifftshift
    2. ifft2
    3. real
    and then mix channels and create an RGB image and return it

    :param mix_channels: a list that contains channels of mixed near and far in frequency domain
    :param width: width of images
    :param height: height of images
    :return: mixed image in spatial domain (final result)
    """
    spatial_channels = []
    for channel in mix_channels:
        mixed_shifted_channel = np.fft.ifftshift(channel)
        mixed_reversed_channel = np.fft.ifft2(mixed_shifted_channel)
        mixed_real_channel = np.real(mixed_reversed_channel)

        spatial_channels.append(mixed_real_channel)

    mix_spatial_image = np.zeros((height, width, 3))
    mix_spatial_image[:, :, 0] = spatial_channels[0]
    mix_spatial_image[:, :, 1] = spatial_channels[1]
    mix_spatial_image[:, :, 2] = spatial_channels[2]

    return mix_spatial_image
