'''create dataset and dataloader'''
import logging
from re import split
import torch.utils.data


def create_dataloader(dataset, dataset_opt, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['num_workers'],
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(dataset_opt, phase, degrade=None):
    '''create dataset'''
    mode = dataset_opt['mode']
    # from data.LRHR_dataset import LRHRDataset as D
    from data.mix_dataset import RestoreDataset as D
    if degrade == 'blur':
        dataset = D(dataroot=dataset_opt['dataroot_blur'],
                    datatype=dataset_opt['datatype'],
                    degrade_type=degrade,
                    patch_size=dataset_opt['patch_size'],
                    # r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LQ=(mode == 'lqhq')
                    )
    elif degrade == 'rain':
        dataset = D(dataroot=dataset_opt['dataroot_rain'],
                    datatype=dataset_opt['datatype'],
                    degrade_type=degrade,
                    patch_size=dataset_opt['patch_size'],
                    # r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LQ=(mode == 'lqhq')
        )
    elif degrade == 'noise':
        dataset = D(dataroot=dataset_opt['dataroot_noise'],
                    datatype=dataset_opt['datatype'],
                    degrade_type=degrade,
                    patch_size=dataset_opt['patch_size'],
                    # r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LQ=(mode == 'lqhq')
                    )
    elif degrade == 'lowlight':
        dataset = D(dataroot=dataset_opt['dataroot_light'],
                    datatype=dataset_opt['datatype'],
                    degrade_type=degrade,
                    patch_size=dataset_opt['patch_size'],
                    # r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_LQ=(mode == 'lqhq')
                    )
    elif not degrade:
        dataset = D(dataroot=dataset_opt['dataroot'],
                    datatype=dataset_opt['datatype'],
                    patch_size=dataset_opt['patch_size'],
                    # r_resolution=dataset_opt['r_resolution'],
                    split=phase,
                    data_len=dataset_opt['data_len'],
                    need_lq=(mode == 'lqhq')
                    )
    else:
        raise NotImplementedError('Degrade type [{:s}] is not included'.format(degrade))
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s} - {:s}] is created with {:d} samples'.format(dataset.__class__.__name__,
                                                           dataset_opt['name'], degrade, len(dataset)))
    return dataset
