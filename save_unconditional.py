import argparse, os, random, torch
from omegaconf import OmegaConf
from taming_comb.modules.style_encoder.network import *
from taming_comb.modules.diffusionmodules.model import * 
from taming_comb.models.cond_transformer import * 
from dataset import dataset_single_enc_sty
from utils import gen_uncond_indices, save_tensor


torch.cuda.empty_cache()

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='0',
                    help="specify the GPU(s)",
                    type=str)

    parser.add_argument("--root_dir", default='/eva_data0/dataset/summer2winter_yosemite',
                    help="dataset path",
                    type=str)

    parser.add_argument("--dataset", default='summer2winter_yosemite',
                    help="dataset directory name",
                    type=str)

    parser.add_argument("--first_stage_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_settingc_256_final_test/settingc_latest.pt',
                    help="first stage model",
                    type=str)

    parser.add_argument("--transformer_model", default='/eva_data7/VQ-I2I/summer2winter_yosemite_512_512_transformer_final_test/n_700.pt',
                    help="transformer model (second stage model)",
                    type=str)

    parser.add_argument("--save_name", default='./summer2winter_yosemite_uncond_gen',
                    help="save directory name",
                    type=str)

    parser.add_argument("--sample_num", default=5,
                    help="the total generation number",
                    type=int)

    parser.add_argument("--sty_domain", default='A',
                    choices=['A', 'B'],
                    help="the domain of unconditional generation (A or B)",
                    type=str)

    parser.add_argument("--ne", default=512,
                    help="the number of embedding",
                    type=int)

    parser.add_argument("--ed", default=512,
                    help="embedding dimension",
                    type=int)

    parser.add_argument("--z_channel",default=256,
                    help="z channel",
                    type=int)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    # load first stage + second stage model
    transformer_config = OmegaConf.load('transformer.yaml')
    transformer_config.model.params.f_path = args.first_stage_model
    transformer_config.model.params.first_stage_model_config.params.embed_dim = args.ed
    transformer_config.model.params.transformer_config.params.vocab_size = args.ne
    transformer_config.model.params.transformer_config.params.n_embd = args.ne
    transformer_config.model.params.cond_stage_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.params.n_embed = args.ne
    transformer_config.model.params.first_stage_model_config.params.ddconfig.z_channels = args.z_channel
    transformer_config.model.params.device = str(device)
    model = instantiate_from_config(transformer_config.model)
    if(os.path.isfile(args.transformer_model)):
        print('load ' + args.transformer_model)
        ck = torch.load( args.transformer_model, map_location=device)
        model.load_state_dict(ck['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    print('Finish Loading!')
    
    os.makedirs(args.save_name, exist_ok=True)

    test_set = dataset_single_enc_sty(args.root_dir, 'test', args.sty_domain, model.first_stage_model, device)

    for i in range(args.sample_num):

        content_idx = gen_uncond_indices(model, device, target_code_size=16, codebook_size=args.ne)

        style_ref_img = test_set[random.randint(0, len(test_set)-1)]

        test_samples = model.decode_to_img(content_idx, 
                              (1, args.ne, content_idx.shape[1], content_idx.shape[2]),
                              style_ref_img['style'], style_ref_img['label'])

        save_tensor(test_samples, args.save_name, '{}_{}'.format(i, style_ref_img['img_name']))