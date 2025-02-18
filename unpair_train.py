import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import torch.nn.functional as F
from dataset import dataset_combine, dataset_unpair
from torch.utils.data import DataLoader
import os
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 
from torch.utils.tensorboard import SummaryWriter
from utils import save_tensor
import argparse

from torch.cuda.amp import autocast as autocast

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='C:/Users/88697/桌面/VQ-I2I/rain100/',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='rain100',
                    help="dataset directory name",
                    type=str)
                    
    parser.add_argument("--ne", default=1024,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=256,
                    help="z channel",
                    type=int)
    

    parser.add_argument("--epoch_start", default=1,
                    help="start from",
                    type=int)

    parser.add_argument("--epoch_end", default=150,
                    help="end at",
                    type=int)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # ONLY MODIFY SETTING HERE
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    batch_size = 1 # 128
    learning_rate = 1e-4       # 256/512 lr=4.5e-6 from 71 epochs
    img_size = 300
    switch_weight = 0.1 # self-reconstruction : a2b/b2a = 10 : 1
    
    
    save_path = '{}_{}_{}_settingc_{}_final_test'.format(args.dataset, args.ed, args.ne, img_size)    # model dir
    print(save_path)

    # load data
    train_data = dataset_unpair(args.root_dir, 'train', 'A', 'B', img_size, img_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)


    f = os.path.join(os.getcwd(), save_path, 'settingc_latest.pt')
    config = OmegaConf.load('config_comb.yaml')
    config.model.target = 'taming_comb.models.vqgan.VQModelCrossGAN_ADAIN'
    config.model.base_learning_rate = learning_rate
    config.model.params.embed_dim = args.ed
    config.model.params.n_embed = args.ne
    config.model.z_channels = args.z_channel
    config.model.resolution = 256
    model = instantiate_from_config(config.model)
    if(os.path.isfile(f)):
        print('load ' + f)
        ck = torch.load(f, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model = model.cuda()
    model.train()

    
    opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                list(model.decoder_a.parameters())+
                                list(model.decoder_b.parameters())+
                                list(model.quantize.parameters())+
                                list(model.quant_conv.parameters())+
                                list(model.post_quant_conv.parameters()),
                                #list(model.style_enc_a.parameters())+
                                #list(model.style_enc_b.parameters())+
                                #list(model.mlp_a.parameters())+
                                #list(model.mlp_b.parameters()),
                                lr=learning_rate, betas=(0.5, 0.999))
    
    #opt_disc_a = torch.optim.Adam(model.loss_a.discriminator.parameters(),
     #                           lr=learning_rate, betas=(0.5, 0.999))
    
    #opt_disc_b = torch.optim.Adam(model.loss_b.discriminator.parameters(),
    #                            lr=learning_rate, betas=(0.5, 0.999))

    if(os.path.isfile(f)):
        print('load ' + f)
        opt_ae.load_state_dict(ck['opt_ae_state_dict'])
        #opt_disc_a.load_state_dict(ck['opt_disc_a_state_dict'])
        #opt_disc_b.load_state_dict(ck['opt_disc_b_state_dict'])


    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)
    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), save_path))
    save_img = os.path.join(os.getcwd(), save_path,'Image')
    os.mkdir(save_img)

    train_ae_a_error = []
    train_ae_b_error = []
    train_disc_a_error = []
    train_disc_b_error = []
    train_disc_a2b_error = []
    train_disc_b2a_error = []
    train_res_rec_error = []
    
    #train_style_a_loss = []
    #train_style_b_loss = []
    #train_content_a_loss = []
    #train_content_b_loss = []

    iterations = len(train_data) // batch_size
    iterations = iterations + 1 if len(train_data) % batch_size != 0 else iterations
    
    
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    accumulation_steps = 24
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(args.epoch_start, args.epoch_end+1):
        for i in range(iterations):

            dataA, dataB = next(iter(train_loader))
            dataA, dataB = dataA.to(device), dataB.to(device)
            
            
            ## Discriminator A
            #opt_disc_a.zero_grad()
            
            #s_a = model.encode(dataA, label=1)
            
            #fakeA, _, _ = model(dataB, label=0, cross=True,s_given=s_a)
            
            
            #recA, qlossA, _ = model(dataA, label=1, cross=False)
            
            """ #b2a_loss, log = model.loss_a( _, dataA,fakeA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")

            a_rec_d_loss, _ = model.loss_a( _, dataA, recA, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")
            
            disc_a_loss = b2a_loss + 0.2*a_rec_d_loss
            disc_a_loss.backward()
            opt_disc_a.step()
            
            
            ## Discriminator B
            opt_disc_b.zero_grad()
            
            s_b = model.encode(dataB, label=0)
            fakeB, _, s_b_sampled = model(dataA, label=1, cross=True,s_given=s_b)

            recB, qlossB, _ = model(dataB, label=0, cross=False)
            
            a2b_loss, log = model.loss_b(_, dataB,fakeB, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train")

            b_rec_d_loss, _ = model.loss_b( _, dataB, recB, optimizer_idx=1, global_step=epoch,
                                    last_layer=None, split="train") """
            
          
            """ disc_b_loss = a2b_loss + 0.2*b_rec_d_loss
            disc_b_loss.backward()
            opt_disc_b.step() """

            
            ## Generator 
            #opt_ae.zero_grad()
                
            # A reconstruction
            with torch.cuda.amp.autocast():

                recA, qlossA = model(dataA, label=1, cross=False)

                aeloss_a, _ = model.loss_a(qlossA, dataA, recA, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=1), split="train")
                                    
                aeloss_a = aeloss_a.to(device)
            
            # cross path with style a
            #AtoBtoA, _, s_a_from_cross = model(fakeA, label=1, cross=False)
            
            # style_a loss
            #style_a_loss = torch.mean(torch.abs(s_a.detach() - s_a_from_cross)).to(device)
            
            # content_b loss
            #c_b_from_cross, _ = model.encode_content(fakeA)
            #_, quant_c_b = model.encode_content(dataB)
            #content_b_loss = torch.mean(torch.abs(quant_c_b.detach() - c_b_from_cross)).to(device)
            
            
            # B reconstruction

                recB, qlossB = model(dataB, label=0, cross=False)

                aeloss_b, _ = model.loss_b(qlossB, dataB, recB, switch_weight=switch_weight, optimizer_idx=0, global_step=epoch,
                                    last_layer=model.get_last_layer(label=0), split="train")
            
                aeloss_b = aeloss_b.to(device)

            
                writer.add_scalar("lossA", aeloss_a.detach(), i)
                writer.add_scalar("lossB", aeloss_b.detach(), i)
                writer.add_scalar("total_loss", aeloss_a.detach()+aeloss_b.detach(), i)

                gen_loss = aeloss_a + aeloss_b  #+ 1.0*(style_a_loss + style_b_loss) # + 0.2*(content_a_loss + content_b_loss)
            
            
                gen_loss = (aeloss_a + aeloss_b) /accumulation_steps  #+ 1.0*(style_a_loss + style_b_loss) # + 0.2*(content_a_loss + content_b_loss)

                scaler.scale(gen_loss).backward()
                if (i+1) % accumulation_steps == 0:
                    scaler.step(opt_ae)                   # 更新参数
                    scaler.update()
                    opt_ae.zero_grad()


            # compute mse loss b/w input and reconstruction
            data = torch.cat((dataA, dataB), 0).to(device)
            rec = torch.cat((recA, recB), 0).to(device)
            recon_error = F.mse_loss( data, rec)

            train_res_rec_error.append(recon_error.item())
            train_ae_a_error.append(aeloss_a.item())
            train_ae_b_error.append(aeloss_b.item())
            #train_disc_a_error.append(disc_a_loss.item())
            #train_disc_b_error.append(disc_b_loss.item())
            #train_disc_a2b_error.append(a2b_loss.item())
            #train_disc_b2a_error.append(b2a_loss.item())
            
            #train_style_a_loss.append(style_a_loss.item())
            #train_style_b_loss.append(style_b_loss.item())
            
            #train_content_a_loss.append(content_a_loss.item())
            #train_content_b_loss.append(content_b_loss.item())


            if (i+1) % 10 == 0:
                save_tensor(dataA, save_img, 'inputA_{}.jpg'.format(i))
                save_tensor(dataB, save_img, 'inputB_{}.jpg'.format(i))
                save_tensor(recA, save_img, 'recA_{}.jpg'.format(i))
                save_tensor(recB, save_img, 'recB_{}.jpg'.format(i))


                _rec  = 'epoch {}, {} iterations\n'.format(epoch, i+1)
                _rec += '(A domain) ae_loss: {:8f}\n'.format(
                            np.mean(train_ae_a_error[-10:]))
                _rec += '(B domain) ae_loss: {:8f}\n'.format(
                            np.mean(train_ae_b_error[-10:]))
                #_rec += 'A vs A2B loss: {:8f}, B vs B2A loss: {:8f}\n'.format(
                #            np.mean(train_disc_a2b_error[-1000:]), np.mean(train_disc_b2a_error[-1000:]))
                _rec += 'recon_error: {:8f}\n\n'.format(
                    np.mean(train_res_rec_error[-10:]))
                
                #_rec += 'style_a_loss: {:8f}\n\n'.format(
                #    np.mean(train_style_a_loss[-1000:]))
                #_rec += 'style_b_loss: {:8f}\n\n'.format(
                #    np.mean(train_style_b_loss[-1000:]))
                
                #_rec += 'content_a_loss: {:8f}\n\n'.format(
                #    np.mean(train_content_a_loss[-1000:]))
                #_rec += 'content_b_loss: {:8f}\n\n'.format(
                #    np.mean(train_content_b_loss[-1000:]))
                
                print(_rec)
                with open(os.path.join(os.getcwd(), save_path, 'loss.txt'), 'a') as f:
                    f.write(_rec)
                    f.close()

        torch.save(
            {
                'model_state_dict': model.state_dict(),
                'opt_ae_state_dict': opt_ae.state_dict(),
                #'opt_disc_a_state_dict': opt_disc_a.state_dict(),
                #'opt_disc_b_state_dict': opt_disc_b.state_dict()
            }, os.path.join(os.getcwd(), save_path, 'settingc_latest.pt'))


        if(epoch % 15 == 0 and epoch >= 15):
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'opt_ae_state_dict': opt_ae.state_dict(),
                    #'opt_disc_a_state_dict': opt_disc_a.state_dict(),
                    #'opt_disc_b_state_dict': opt_disc_b.state_dict()
                }, os.path.join(os.getcwd(), save_path, 'settingc_n_{}.pt'.format(epoch)))