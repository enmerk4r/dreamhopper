from flask import Flask, request
import redis
from uuid import uuid4
import json
from threading import Thread
from PIL import Image
import base64
from io import BytesIO

import torch
import random

from nerf.provider import NeRFDataset
from nerf.network import NeRFNetwork
from nerf.utils import *
from options import Options

app = Flask(__name__)

@app.route("/generate", methods=['POST'])
def generate():
    data = request.get_json()

    text = data["text"]
    iters = data["iters"]
    seed = data["seed"]
    w = data["w"]
    h = data["h"]
    W = data["W"]
    H = data["H"]

    if seed == -1:
        seed = random.sample(range(0,np.iinfo(np.int32).max),1)[0]
    seed_everything(seed)

    opt = Options(text, seed, w=w, h=h, W=W, H=H, iterations=iters)

    generate_mesh(opt)

    



    id = str(uuid4())
    return {
        "id": id
    }

def generate_mesh(opt):
    model = NeRFNetwork(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    # get the camera parameters here
    train_loader = NeRFDataset(opt, device=device, type='train', H=opt.h, W=opt.w, radius=opt.radius, fovy=opt.fovy, size=100).dataloader()

    # decay to 0.1 * init_lr at last iter step
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.val_int)

    valid_loader = NeRFDataset(opt, device=device, type='val', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=opt.val_samples).dataloader()
    max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
    trainer.train(train_loader, valid_loader, max_epoch)

    # also test
    test_loader = NeRFDataset(opt, device=device, type='test', H=opt.H, W=opt.W, radius=opt.radius, fovy=opt.fovy, size=opt.test_samples).dataloader()
    trainer.save_mesh(resolution=opt.mesh_res, threshold=opt.mesh_trh)
    trainer.test(test_loader, write_video=False) # test and save video

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1984)