class Options():
    def __init__(
        self, 
        text,
        seed,
        iterations=2000,
        w=160,
        h=160,
        W=160,
        H=160,
        lr=0.0005
        ):
        self.text=text
        self.image=None
        self.image_direction=None
        self.workspace='pavilion_1106-0010'
        self.output_dir='./output'
        self.seed=seed
        self.colab=False
        self.val_int=10
        self.save_interval_img=False
        self.val_samples=12
        self.save_depth=False
        self.iters=iterations
        self.lr=lr
        self.ckpt='latest'
        self.ckpt_save_interval=50
        self.num_rays=4096
        self.cuda_ray=True
        self.num_steps=512
        self.upsample_steps=0
        self.max_ray_batch=4096
        self.clip_model='ViT-L/14'
        self.clip_aug=True
        self.rnd_fovy=True
        self.test=False
        self.save_video=False
        self.save_mesh=False
        self.save_density_map=False
        self.test_samples=12
        self.mesh_res=256
        self.mesh_trh=10
        self.fp16=True
        self.bound=1
        self.dt_gamma=0.0
        self.w=w
        self.h=h
        self.gui=False
        self.W=W
        self.H=H
        self.radius=3
        self.fovy=43.0
        self.max_spp=64
        self.tau_0=0.5
        self.tau_1=0.8
        self.tau_step=500
        self.aug_copy=8
        self.dir_text=True

    def toDict(self):
        return {
            "text" : self.text,
            "image" : self.image,
            "image_direction": self.image_direction,
            "workspace": self.workspace,
            "output_dir": self.output_dir,
            "seed": self.seed,
            "colab": self.colab,
            "val_int": self.val_int,
            "save_interval_img": self.save_interval_img,
            "val_samples": self.val_samples,
            "save_depth": self.save_depth,
            "iters": self.iters,
            "lr": self.lr,
            "ckpt": self.ckpt,
            "ckpt_save_interval" : self.ckpt_save_interval,
            "num_rays": self.num_rays,
            "cuda_ray": self.cuda_ray,
            "num_steps": self.num_steps,
            "upsample_steps": self.upsample_steps,
            "max_ray_batch": self.max_ray_batch,
            "clip_model":self.clip_model,
            "clip_aug":self.clip_aug,
            "rnd_fovy":self.rnd_fovy,
            "test":self.test,
            "save_video":self.save_video,
            "save_mesh":self.save_mesh,
            "save_density_map":self.save_density_map,
            "test_samples":self.test_samples,
            "mesh_res":self.mesh_res,
            "mesh_trh":self.mesh_trh,
            "fp16":self.fp16,
            "bound": self.bound,
            "dt_gamma": self.dt_gamma,
            "w": self.w,
            "h" : self.h,
            "gui" : self.gui,
            "W" : self.W,
            "h": self.H,
            "radius" : self.radius,
            "fovy" : self.fovy,
            "max_spp" : self.max_spp,
            "tau_0" : self.tau_0,
            "tau_1" : self.tau_1,
            "tau_step" : self.tau_step,
            "aug_copy" : self.aug_copy,
            "dir_text" : self.dir_text
        }