import torch
import argparse
import random

from nerf.provider import NeRFDataset
from nerf.utils import *

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--image', default=None, help="ref image prompt")
    parser.add_argument('--image_direction', default=None, help="the view direction if the image prompt: 'front', 'left side', 'back', 'right side', 'top', 'bottom'")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--seed', type=int, default=-1,help="set to -1 to get random seed")
    ### colab options
    parser.add_argument('--colab', action='store_true', help="running in colab notebook mode")
    parser.add_argument('--val_int', type=int, default=20, help="interval of saving and displaying intermediate output")
    parser.add_argument('--save_interval_img', action='store_true', help="if save interval images")
    parser.add_argument('--val_samples', type=int, default=10, help="how many samples to use for validation in training")
    parser.add_argument('--save_depth', action='store_true', help="if save interval images")
    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters, the final epoch wil be divied by 100")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--ckpt_save_interval', type=int, default=50)
    parser.add_argument('--num_rays', type=int, default=4096)
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--clip_model', type=str, default='ViT-B/16',help="the clip applied")
    parser.add_argument('--clip_aug', action='store_true', help="use random augmentation for the render iamge before feed into clip")
    parser.add_argument('--rnd_fovy', action='store_true', help="use random fovy (view angle in traning")
     ### test options
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_video', action='store_true', help="save video in testing")
    parser.add_argument('--save_mesh', action='store_true', help="save video in testing")
    parser.add_argument('--save_density_map', action='store_true', help="save density map in testing and valiation")
    parser.add_argument('--test_samples', type=int, default=12, help="set the total frames of video output")
    parser.add_argument('--mesh_res', type=int, default=256, help="the resolution of mesh grid")
    parser.add_argument('--mesh_trh', type=int, default=10, help="the threshold of marching cube")
    ## network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    #TensoRF curently not available
    # parser.add_argument('--cc', action='store_true', help="use TensoRF")
    ### dataset options
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--w', type=int, default=128, help="render width for CLIP training (significantly affect the GPU RAM ocupy)")
    parser.add_argument('--h', type=int, default=128, help="render height for CLIP training (significantly affect the GPU RAM ocupy)")
    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=3, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=90, help="default GUI camera fovy (view angle)")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")
    ### other options
    parser.add_argument('--tau_0', type=float, default=0.5, help="target mean transparency 0")
    parser.add_argument('--tau_1', type=float, default=0.8, help="target mean transparency 1")
    parser.add_argument('--tau_step', type=float, default=500, help="steps to anneal from tau_0 to tau_1")
    parser.add_argument('--aug_copy', type=int, default=8, help="augmentation copy for each renderred image before feeding into CLIP")
    parser.add_argument('--dir_text', action='store_true', help="direction encoded text prompt")

    opt = parser.parse_args()

    assert not (opt.text is None and opt.image is None)
    

    # if opt.cc:
    #     from nerf.network_cc import NeRFNetwork
    # else:
    from nerf.network import NeRFNetwork

    print(opt)
    if opt.seed == -1:
        opt.seed = random.sample(range(0,np.iinfo(np.int32).max),1)[0]
    seed_everything(opt.seed)

    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
    )

    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint='latest')

        if opt.gui:
            from nerf.gui import NeRFGUI
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=opt.test_samples).dataloader()
            if opt.save_mesh:
                trainer.save_mesh(resolution=opt.mesh_res, threshold=opt.mesh_trh)
            trainer.test(test_loader, write_video=opt.save_video) # test and save video
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # get the camera parameters here
        train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, radius=opt.radius, fovy=opt.fovy, size=100).dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.val_int)

        if opt.gui:
            from nerf.gui import NeRFGUI
            trainer.train_loader = train_loader # attach dataloader to trainer

            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=opt.val_samples).dataloader()
            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=opt.test_samples).dataloader()
            trainer.save_mesh(resolution=opt.mesh_res, threshold=opt.mesh_trh)
            trainer.test(test_loader, write_video=opt.save_video) # test and save video