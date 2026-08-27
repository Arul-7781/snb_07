'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data

def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        if sampler is not None:
            shuffle = False

        pin_memory = bool(dataset_opt.get('pin_memory', True))
        persistent_workers = bool(dataset_opt.get('persistent_workers', num_workers > 0))
        prefetch_factor = dataset_opt.get('prefetch_factor', 2)
        drop_last = bool(dataset_opt.get('drop_last', True))

        kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'sampler': sampler,
            'drop_last': drop_last,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers if num_workers > 0 else False,
        }
        if num_workers > 0 and prefetch_factor is not None:
            kwargs['prefetch_factor'] = int(prefetch_factor)

        return torch.utils.data.DataLoader(dataset, **kwargs)
    else:
        num_workers = int(dataset_opt.get('n_workers', 1))
        pin_memory = bool(dataset_opt.get('pin_memory', True))
        persistent_workers = bool(dataset_opt.get('persistent_workers', num_workers > 0))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'test':
        from data.video_test_dataset import VideoTestDataset as D
    elif mode == 'train':
        from data.Vimeo90K_dataset import Vimeo90KDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    print(mode)
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
