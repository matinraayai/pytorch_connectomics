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


def build_train_augmentor(model_input_size, do_2d, keep_uncropped=False, keep_non_smoothed=False, **kwargs):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []
    names = kwargs['ADDITIONAL_TARGETS_NAME']
    types = kwargs['ADDITIONAL_TARGETS_TYPE']
    if names is None:
        additional_targets = None
    else:
        assert len(names) == len(types)
        additional_targets = {}
        for i in range(len(names)):
            additional_targets[names[i]] = types[i]

    # 1. rotate
    if kwargs['ROTATE']['ENABLED']:
        aug_list.append(Rotate(rot90=kwargs['ROTATE']['ROT90'],
                        p=kwargs['ROTATE']['P'],
                        additional_targets=additional_targets))

    # 2. rescale
    if kwargs['RESCALE']['ENABLED']:
        aug_list.append(Rescale(p=kwargs['RESCALE']['P'],
                        additional_targets=additional_targets))

    # 3. flip
    if kwargs['FLIP']['ENABLED']:
        aug_list.append(Flip(do_ztrans=kwargs['FLIP']['DO_ZTRANS'],
                        p=kwargs['FLIP']['P'],
                        additional_targets=additional_targets))

    # 4. elastic
    if kwargs['ELASTIC']['ENABLED']:
        aug_list.append(Elastic(alpha=kwargs['ELASTIC']['ALPHA'],
                        sigma=kwargs['ELASTIC']['SIGMA'],
                        p=kwargs['ELASTIC']['P'],
                        additional_targets=additional_targets))

    # 5. grayscale
    if kwargs['GRAYSCALE']['ENABLED']:
        aug_list.append(Grayscale(p=kwargs['GRAYSCALE']['P'],
                        additional_targets=additional_targets))

    # 6. missing parts
    if kwargs['MISSINGPARTS']['ENABLED']:
        aug_list.append(MissingParts(p=kwargs['MISSINGPARTS']['P'],
                        additional_targets=additional_targets))

    # 7. missing section
    if kwargs['MISSINGSECTION']['ENABLED'] and not do_2d:
        aug_list.append(MissingSection(num_sections=kwargs['MISSINGSECTION']['NUM_SECTION'],
                                       p=kwargs['MISSINGSECTION']['P'],
                                       additional_targets=additional_targets))

    # 8. misalignment
    if kwargs['MISALIGNMENT']['ENABLED'] and not do_2d:
        aug_list.append(MisAlignment(displacement=kwargs['MISALIGNMENT']['DISPLACEMENT'],
                                     rotate_ratio=kwargs['MISALIGNMENT']['ROTATE_RATIO'],
                                     p=kwargs['MISALIGNMENT']['P'],
                                     additional_targets=additional_targets))

    # 9. motion-blur
    if kwargs['MOTIONBLUR']['ENABLED']:
        aug_list.append(MotionBlur(sections=kwargs['MOTIONBLUR']['SECTIONS'],
                                   kernel_size=kwargs['MOTIONBLUR']['KERNEL_SIZE'],
                                   p=kwargs['MOTIONBLUR']['P'],
                                   additional_targets=additional_targets))

    # 10. cut-blur
    if kwargs['CUTBLUR']['ENABLED']:
        aug_list.append(CutBlur(length_ratio=kwargs['CUTBLUR']['LENGTH_RATIO'],
                                down_ratio_min=kwargs['CUTBLUR']['DOWN_RATIO_MIN'],
                                down_ratio_max=kwargs['CUTBLUR']['DOWN_RATIO_MAX'],
                                downsample_z=kwargs['CUTBLUR']['DOWNSAMPLE_Z'],
                                p=kwargs['CUTBLUR']['P'],
                                additional_targets=additional_targets))

    #11. cut-noise
    if kwargs['CUTNOISE']['ENABLED']:
        aug_list.append(CutNoise(length_ratio=kwargs['CUTNOISE']['LENGTH_RATIO'],
                        scale=kwargs['CUTNOISE']['SCALE'],
                        p=kwargs['CUTNOISE']['P'],
                        additional_targets=additional_targets))

    augmentor = Compose(transforms=aug_list, 
                        input_size=model_input_size,
                        smooth=kwargs['SMOOTH'],
                        keep_uncropped=keep_uncropped, 
                        keep_non_smoothed=keep_non_smoothed,
                        additional_targets=additional_targets)
    return augmentor
