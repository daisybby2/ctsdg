import os

import torch
import numpy as np
from tqdm import tqdm

from tensorboardX import SummaryWriter

from utils.distributed import get_rank, reduce_loss_dict
from utils.misc import requires_grad, sample_data
from criteria.loss import generator_loss_func, discriminator_loss_func


def train(opts, image_data_loader, generator, discriminator, extractor, generator_optim, discriminator_optim, is_cuda):

    image_data_loader = sample_data(image_data_loader)#加载样本图像，进行训练
    pbar = range(opts.train_iter)#迭代训练的迭代器
    if get_rank() == 0:#在控制台打印出进度条
        pbar = tqdm(pbar, initial=opts.start_iter, dynamic_ncols=True, smoothing=0.01)
    
    if opts.distributed:#if else语句判断当前环境是否是分布式训练环境，根据具体情况加载模型
        generator_module, discriminator_module = generator.module, discriminator.module
    else:
        generator_module, discriminator_module = generator, discriminator
    
    writer = SummaryWriter(opts.log_dir)#训练过程中的信息写入tensorboard

    for index in pbar:#应该是，一轮=1个batch_size，一次循环体
        
        i = index + opts.start_iter#计算训练的轮数，因为存在中断后继续训练的情况，所以start不一定是0
        if i > opts.train_iter: #训练达到最大轮数，结束
            print('Done...')
            break

        ground_truth, mask, edge, gray_image = next(image_data_loader)#获取下一batch的数据

        if is_cuda:
            ground_truth, mask, edge, gray_image = ground_truth.cuda(), mask.cuda(), edge.cuda(), gray_image.cuda()

        input_image, input_edge, input_gray_image = ground_truth * mask, edge * mask, gray_image * mask#将完成样本信息进行掩膜，生成空洞。

        # ---------
        # Generator
        # ---------
        requires_grad(generator, True)
        requires_grad(discriminator, False)#在GAN中，生成器和判别器分开训练，因此训练生成器时，需要将判别器的梯度计算设置为false，保证反向传播的过程只更新生成器

        output, projected_image, projected_edge = generator(input_image, torch.cat((input_edge, input_gray_image), dim=1), mask)
        comp = ground_truth * mask + output * (1 - mask)#提取生成结果和gt拼接成的完成图像
  
        output_pred, output_edge = discriminator(output, gray_image, edge, is_real=False)#输出图像判别结果和提取出来的边缘信息（并不是canny)
        
        vgg_comp, vgg_output, vgg_ground_truth = extractor(comp), extractor(output), extractor(ground_truth)#使用VGG16从混合结果、生成器结果和GT中提取特征

        generator_loss_dict = generator_loss_func(#生成器的loss字典
            mask, output, ground_truth, edge, output_pred, 
            vgg_comp, vgg_output, vgg_ground_truth, 
            projected_image, projected_edge,
            output_edge
        )
        generator_loss = generator_loss_dict['loss_hole'] * opts.HOLE_LOSS + \
                         generator_loss_dict['loss_valid'] * opts.VALID_LOSS + \
                         generator_loss_dict['loss_perceptual'] * opts.PERCEPTUAL_LOSS + \
                         generator_loss_dict['loss_style'] * opts.STYLE_LOSS + \
                         generator_loss_dict['loss_adversarial'] * opts.ADVERSARIAL_LOSS + \
                         generator_loss_dict['loss_intermediate'] * opts.INTERMEDIATE_LOSS
        generator_loss_dict['loss_joint'] = generator_loss#按照配置文件中的权重，计算整个生成器的联合损失

        #以下三行完成对生成器的参数更新
        generator_optim.zero_grad()#清空生成器的梯度，防止梯度叠加
        generator_loss.backward()#反向传播
        generator_optim.step()#参数更新

        # -------------
        # Discriminator
        # -------------
        requires_grad(generator, False)
        requires_grad(discriminator, True)#固定生成器，训练判别器

        real_pred, real_pred_edge = discriminator(ground_truth, gray_image, edge, is_real=True)#真实图像判别
        fake_pred, fake_pred_edge = discriminator(output.detach(), gray_image, edge, is_real=False)#生成图像判别

        discriminator_loss_dict = discriminator_loss_func(real_pred, fake_pred, real_pred_edge, fake_pred_edge, edge)
        discriminator_loss = discriminator_loss_dict['loss_adversarial']
        discriminator_loss_dict['loss_joint'] = discriminator_loss#判别器的对抗损失

        discriminator_optim.zero_grad()#这三行更新判别器的参数
        discriminator_loss.backward()
        discriminator_optim.step()

        # ---
        # log
        # ---
        generator_loss_dict_reduced, discriminator_loss_dict_reduced = reduce_loss_dict(generator_loss_dict), reduce_loss_dict(discriminator_loss_dict)
        #把多个GPU上的损失平均在一起
        pbar_g_loss_hole = generator_loss_dict_reduced['loss_hole'].mean().item()
        pbar_g_loss_valid = generator_loss_dict_reduced['loss_valid'].mean().item()
        pbar_g_loss_perceptual = generator_loss_dict_reduced['loss_perceptual'].mean().item()
        pbar_g_loss_style = generator_loss_dict_reduced['loss_style'].mean().item()
        pbar_g_loss_adversarial = generator_loss_dict_reduced['loss_adversarial'].mean().item()
        pbar_g_loss_intermediate = generator_loss_dict_reduced['loss_intermediate'].mean().item()
        pbar_g_loss_joint = generator_loss_dict_reduced['loss_joint'].mean().item()

        pbar_d_loss_adversarial = discriminator_loss_dict_reduced['loss_adversarial'].mean().item()#判别器的对抗损失
        pbar_d_loss_joint = discriminator_loss_dict_reduced['loss_joint'].mean().item()#判别器的联合损失

        if get_rank() == 0:

            pbar.set_description((
                f'g_loss_joint: {pbar_g_loss_joint:.4f} '
                f'd_loss_joint: {pbar_d_loss_joint:.4f}'
            ))

            writer.add_scalar('g_loss_hole', pbar_g_loss_hole, i)
            writer.add_scalar('g_loss_valid', pbar_g_loss_valid, i)
            writer.add_scalar('g_loss_perceptual', pbar_g_loss_perceptual, i)
            writer.add_scalar('g_loss_style', pbar_g_loss_style, i)
            writer.add_scalar('g_loss_adversarial', pbar_g_loss_adversarial, i)
            writer.add_scalar('g_loss_intermediate', pbar_g_loss_intermediate, i)
            writer.add_scalar('g_loss_joint', pbar_g_loss_joint, i)

            writer.add_scalar('d_loss_adversarial', pbar_d_loss_adversarial, i)
            writer.add_scalar('d_loss_joint', pbar_d_loss_joint, i)

            if i % opts.save_interval == 0:#定期保存模型

                torch.save(
                    {
                        'n_iter': i,
                        'generator': generator_module.state_dict(),
                        'discriminator': discriminator_module.state_dict()
                    },
                    f"{opts.save_dir}/{str(i).zfill(6)}.pt",
                )

