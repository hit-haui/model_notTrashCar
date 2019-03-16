from imgaug import augmenters as iaa
import automold as am


class RandomShadow(iaa.meta.Augmenter):

    def __init__(self, abc=1, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(RandomShadow, self).__init__()
        self.abc = abc

    def _augment_images(self, images, random_state, parents, hooks):
        result = images

        for _idx, image in enumerate(images):
            result[_idx] = am.add_shadow(image)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.abc]


class RandomGravel(iaa.meta.Augmenter):

    def __init__(self, abc=1, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(RandomGravel, self).__init__()
        self.abc = abc

    def _augment_images(self, images, random_state, parents, hooks):
        result = images

        for _idx, image in enumerate(images):
            result[_idx] = am.add_gravel(image)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.abc]


class RandomSunFlare(iaa.meta.Augmenter):

    def __init__(self, abc=1, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(RandomSunFlare, self).__init__()
        self.abc = abc

    def _augment_images(self, images, random_state, parents, hooks):
        result = images

        for _idx, image in enumerate(images):
            result[_idx] = am.add_sun_flare(image)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.abc]


class RandomGravel(iaa.meta.Augmenter):

    def __init__(self, abc=1, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(RandomGravel, self).__init__()
        self.abc = abc

    def _augment_images(self, images, random_state, parents, hooks):
        result = images

        for _idx, image in enumerate(images):
            result[_idx] = am.add_gravel(image)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.abc]


def motionblur(img):
    h, w, _ = img.shape
    # left
    augment_blur_left = iaa.MotionBlur(
        k=15, angle=255, direction=-1.0, order=0)
    img_drop_left = img[:h, :int(w/2)]
    motion_left = augment_blur_left.augment_image(img_drop_left)
    # right
    augment_blur_right = iaa.MotionBlur(
        k=15, angle=105, direction=1.0, order=0)
    img_drop_right = img[:h, int(w/2):w]
    motion_right = augment_blur_right.augment_image(img_drop_right)
    ###
    img[:h, :int(w/2)] = motion_left
    img[:h, int(w/2):w] = motion_right
    return img


class RandomMotionBlur(iaa.meta.Augmenter):

    def __init__(self, abc=1, name=None, deterministic=False,
                 random_state=None):
        """Initialize the augmentator"""
        super(RandomMotionBlur, self).__init__()
        self.abc = abc

    def _augment_images(self, images, random_state, parents, hooks):
        result = images

        for _idx, image in enumerate(images):
            result[_idx] = motionblur(image)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents,
                           hooks):
        return keypoints_on_images

    def _augment_heatmaps(self, **kwargs):
        return

    def get_parameters(self):
        return [self.abc]