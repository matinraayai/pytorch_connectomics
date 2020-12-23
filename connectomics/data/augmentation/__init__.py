from .composition import Compose
from .augmentor import DataAugment
from .test_augmentor import TestAugmentor

# augmentation methods
from .warp import Elastic
from .grayscale import Grayscale
from .flip import Flip
from .rotation import Rotate
from .rescale import Rescale
from .misalign import MisAlignment
from .missing_section import MissingSection
from .missing_parts import MissingParts
from .motion_blur import MotionBlur
from .cutblur import CutBlur
from .cutnoise import CutNoise
from .mixup import MixupAugmentor

__all__ = ['Compose',
           'DataAugment', 
           'Elastic',
           'Grayscale',
           'Rotate',
           'Rescale',
           'MisAlignment',
           'MissingSection',
           'MissingParts',
           'Flip',
           'MotionBlur',
           'CutBlur',
           'CutNoise',
           'MixupAugmentor',
           'TestAugmentor']


def build_train_augmentor(keep_uncropped=False, keep_non_smoothed=False, **kwargs):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []
    # 1. rotate
    if kwargs['ROTATE']['ENABLED']:
        aug_list.append(Rotate(p=kwargs['ROTATE']['P']))

    # 2. rescale
    if kwargs['RESCALE']['ENABLED']:
        aug_list.append(Rescale(p=kwargs['RESCALE']['P']))

    # 3. flip
    if kwargs['FLIP']['ENABLED']:
        aug_list.append(Flip(p=kwargs['FLIP']['P'],
                             do_ztrans=kwargs['FLIP']['DO_ZTRANS']))

    # 4. elastic
    if kwargs['ELASTIC']['ENABLED']:
        aug_list.append(Elastic(alpha=kwargs['ELASTIC']['ALPHA'],
                                sigma=kwargs['ELASTIC']['SIGMA'],
                                p=kwargs['ELASTIC']['P']))

    # 5. grayscale
    if kwargs['GRAYSCALE']['ENABLED']:
        aug_list.append(Grayscale(p=kwargs['GRAYSCALE']['P']))

    # 6. missingparts
    if kwargs['MISSINGPARTS']['ENABLED']:
        aug_list.append(MissingParts(p=kwargs['MISSINGPARTS']['P']))

    # 7. missingsection
    if kwargs['MISSINGSECTION']['ENABLED'] and not kwargs['DATASET']['DO_2D']:
            aug_list.append(MissingSection(p=kwargs['MISSINGSECTION']['P'],
                                           num_sections=kwargs['MISSINGSECTION']['NUM_SECTION']))

    # 8. misalignment
    if kwargs['MISALIGNMENT']['ENABLED'] and not kwargs['DATASET']['DO_2D']:
            aug_list.append(MisAlignment(p=kwargs['MISALIGNMENT']['P'],
                                         displacement=kwargs['MISALIGNMENT']['DISPLACEMENT'],
                                         rotate_ratio=kwargs['MISALIGNMENT']['ROTATE_RATIO']))
    # 9. motion-blur
    if kwargs['MOTIONBLUR']['ENABLED']:
        aug_list.append(MotionBlur(p=kwargs['MOTIONBLUR']['P'],
                                   sections=kwargs['MOTIONBLUR']['SECTIONS'],
                                   kernel_size=kwargs['MOTIONBLUR']['KERNEL_SIZE']))

    # 10. cut-blur
    if kwargs['CUTBLUR']['ENABLED']:
        aug_list.append(CutBlur(p=kwargs['CUTBLUR']['P'],
                                length_ratio=kwargs['CUTBLUR']['LENGTH_RATIO'],
                                down_ratio_min=kwargs['CUTBLUR']['DOWN_RATIO_MIN'],
                                down_ratio_max=kwargs['CUTBLUR']['DOWN_RATIO_MAX'],
                                downsample_z=kwargs['CUTBLUR']['DOWNSAMPLE_Z']))

    # 11. cut-noise
    if kwargs['CUTNOISE']['ENABLED']:
        aug_list.append(CutNoise(p=kwargs['CUTNOISE']['P'],
                                 length_ratio=kwargs['CUTNOISE']['LENGTH_RATIO'],
                                 scale=kwargs['CUTNOISE']['SCALE']))

    augmentor = Compose(aug_list, input_size=kwargs['MODEL']['INPUT_SIZE'], smooth=kwargs['AUGMENTOR']['SMOOTH'],
                        keep_uncropped=keep_uncropped, keep_non_smoothed=keep_non_smoothed)

    return augmentor
