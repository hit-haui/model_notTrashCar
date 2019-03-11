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