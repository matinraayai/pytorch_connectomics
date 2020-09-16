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


def build_train_augmentor(cfg, keep_uncropped=False, keep_non_smoothed=False):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []
    # 1. rotate
    if cfg.augmentor.ROTATE.ENABLED:
        aug_list.append(Rotate(p=cfg.augmentor.ROTATE.P))

    # 2. rescale
    if cfg.augmentor.RESCALE.ENABLED:
        aug_list.append(Rescale(p=cfg.augmentor.RESCALE.P))

    # 3. flip
    if cfg.augmentor.FLIP.ENABLED:
        aug_list.append(Flip(p=cfg.augmentor.FLIP.P,
                             do_ztrans=cfg.augmentor.FLIP.DO_ZTRANS))

    # 4. elastic
    if cfg.augmentor.ELASTIC.ENABLED:
        aug_list.append(Elastic(alpha=cfg.augmentor.ELASTIC.ALPHA,
                                sigma = cfg.augmentor.ELASTIC.SIGMA,
                                p=cfg.augmentor.ELASTIC.P))

    # 5. grayscale
    if cfg.augmentor.GRAYSCALE.ENABLED:
        aug_list.append(Grayscale(p=cfg.augmentor.GRAYSCALE.P))

    # 6. missingparts
    if cfg.augmentor.MISSINGPARTS.ENABLED:
        aug_list.append(MissingParts(p=cfg.augmentor.MISSINGPARTS.P))

    # 7. missingsection
    if cfg.augmentor.MISSINGSECTION.ENABLED and not cfg.dataset.do_2D:
            aug_list.append(MissingSection(p=cfg.augmentor.MISSINGSECTION.P,
                                           num_sections=cfg.augmentor.MISSINGSECTION.NUM_SECTION))

    # 8. misalignment
    if cfg.augmentor.MISALIGNMENT.ENABLED and not cfg.dataset.do_2D:
            aug_list.append(MisAlignment(p=cfg.augmentor.MISALIGNMENT.P,
                                         displacement=cfg.augmentor.MISALIGNMENT.DISPLACEMENT,
                                         rotate_ratio=cfg.augmentor.MISALIGNMENT.ROTATE_RATIO))
    # 9. motion-blur
    if cfg.augmentor.MOTIONBLUR.ENABLED:
        aug_list.append(MotionBlur(p=cfg.augmentor.MOTIONBLUR.P,
                                   sections=cfg.augmentor.MOTIONBLUR.SECTIONS,
                                   kernel_size=cfg.augmentor.MOTIONBLUR.KERNEL_SIZE))

    # 10. cut-blur
    if cfg.augmentor.CUTBLUR.ENABLED:
        aug_list.append(CutBlur(p=cfg.augmentor.CUTBLUR.P,
                                length_ratio=cfg.augmentor.CUTBLUR.LENGTH_RATIO,
                                down_ratio_min=cfg.augmentor.CUTBLUR.DOWN_RATIO_MIN,
                                down_ratio_max=cfg.augmentor.CUTBLUR.DOWN_RATIO_MAX,
                                downsample_z=cfg.augmentor.CUTBLUR.DOWNSAMPLE_Z))

    # 11. cut-noise
    if cfg.augmentor.CUTNOISE.ENABLED:
        aug_list.append(CutNoise(p=cfg.augmentor.CUTNOISE.P,
                                 length_ratio=cfg.augmentor.CUTNOISE.LENGTH_RATIO,
                                 scale=cfg.augmentor.CUTNOISE.SCALE))

    augmentor = Compose(aug_list, input_size=cfg.model.input_size, smooth=cfg.augmentor.SMOOTH,
                        keep_uncropped=keep_uncropped, keep_non_smoothed=keep_non_smoothed)

    return augmentor
