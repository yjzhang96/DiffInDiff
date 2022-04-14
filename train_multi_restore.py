import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import wandb
import random 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_blur_set = Data.create_dataset(dataset_opt, phase, 'blur')
            train_blur_loader = Data.create_dataloader(
                train_blur_set, dataset_opt, phase)
            train_rain_set = Data.create_dataset(dataset_opt, phase, 'rain')
            train_rain_loader = Data.create_dataloader(
                train_rain_set, dataset_opt, phase)
            train_noise_set = Data.create_dataset(dataset_opt, phase, 'noise')
            train_noise_loader = Data.create_dataloader(
                train_noise_set, dataset_opt, phase)
            train_light_set = Data.create_dataset(dataset_opt, phase, 'lowlight')
            train_light_loader = Data.create_dataloader(
                train_light_set, dataset_opt, phase)
            train_degrade_num = dataset_opt['degrade_num']
        elif phase == 'val':
            val_blur_set = Data.create_dataset(dataset_opt, phase, 'blur')
            val_blur_loader = Data.create_dataloader(
                val_blur_set, dataset_opt, phase)
            val_rain_set = Data.create_dataset(dataset_opt, phase, 'rain')
            val_rain_loader = Data.create_dataloader(
                val_rain_set, dataset_opt, phase)
            val_noise_set = Data.create_dataset(dataset_opt, phase, 'noise')
            val_noise_loader = Data.create_dataloader(
                val_noise_set, dataset_opt, phase)
            val_light_set = Data.create_dataset(dataset_opt, phase, 'lowlight')
            val_light_loader = Data.create_dataloader(
                train_light_set, dataset_opt, phase)    
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    
    iter_per_epoch = max(len(train_blur_loader),len(train_rain_loader),len(train_noise_loader))
    print('There is {:d} iteration in one epoch:'.format(iter_per_epoch))
    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    # diffusion.set_new_noise_schedule(
    #     opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            blur_data_iter = iter(train_blur_loader)
            rain_data_iter = iter(train_rain_loader)
            noise_data_iter = iter(train_noise_loader)
            light_data_iter = iter(train_light_loader)
            for _ in range(iter_per_epoch):
                random_type = random.randint(0,train_degrade_num-1)
                if random_type == 0:
                    try:
                        train_data = next(blur_data_iter)
                    except:
                        blur_data_iter = iter(train_blur_loader)
                        continue
                elif random_type == 1:
                    try:
                        train_data = next(rain_data_iter)
                    except:
                        rain_data_iter = iter(train_rain_loader)
                        continue 
                elif random_type == 2:
                    try:
                        train_data = next(noise_data_iter) 
                    except:
                        print("The end of noise data iterator")
                        noise_data_iter = iter(train_noise_loader)
                        continue
                elif random_type == 3:
                    try:
                        train_data = next(light_data_iter) 
                    except:
                        print("The end of light data iterator")
                        light_data_iter = iter(train_light_loader)
                        continue
                else:
                    raise TypeError('dataloader type not recognized')

                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0 or current_step==1:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    # diffusion.set_new_noise_schedule(
                    #     opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_blur_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        Restore_img = Metrics.tensor2img(visuals['Restore'])  # uint8
                        HQ_img = Metrics.tensor2img(visuals['HQ'])  # uint8
                        LQ_img = Metrics.tensor2img(visuals['LQ'])  # uint8
                        # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            HQ_img, '{}/{}_{}_HQ.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            Restore_img, '{}/{}_{}_Restore.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            LQ_img, '{}/{}_{}_LQ.png'.format(result_path, current_step, idx))
                        # Metrics.save_img(
                        #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (LQ_img, Restore_img, HQ_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            Restore_img, HQ_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((LQ_img, Restore_img, HQ_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    # diffusion.set_new_noise_schedule(
                    #     opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_blur_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            HQ_img = Metrics.tensor2img(visuals['HQ'])  # uint8
            LQ_img = Metrics.tensor2img(visuals['LQ'])  # uint8
            # fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                Restore_img = visuals['Restore']  # uint8
                sample_num = Restore_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(Restore_img[iter]), '{}/{}_{}_Restore_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                Restore_img = Metrics.tensor2img(visuals['Restore'])  # uint8
                Metrics.save_img(
                    Restore_img, '{}/{}_{}_Restore_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['Restore'][-1]), '{}/{}_{}_Restore.png'.format(result_path, current_step, idx))
                Condition_img = Metrics.tensor2img(visuals['Cond'])
                Metrics.save_img(
                    Condition_img, '{}/{}_{}_Condition_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                HQ_img, '{}/{}_{}_HQ.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                LQ_img, '{}/{}_{}_LQ.png'.format(result_path, current_step, idx))
            # Metrics.save_img(
            #     fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['Restore'][-1]), HQ_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['Restore'][-1]), HQ_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(LQ_img, Metrics.tensor2img(visuals['Restore'][-1]), HQ_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
