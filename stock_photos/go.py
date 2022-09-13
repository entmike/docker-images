ckpt = "/weights/sd-v1-4.ckpt"
config = "/workspace/k-diffusion/v1-inference.yaml"

import os, sys
sys.path.append("/workspace/k-diffusion")

# Environment Vars
from dotenv import load_dotenv
load_dotenv()

WORKER = os.getenv('WORKER')
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
MONGODB_CONNECTION = os.getenv('MONGODB_CONNECTION')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')

print(f"Worker name: {WORKER}")

from IPython.display import Image as Img
from types import SimpleNamespace
from pymongo import UpdateOne
import boto3
from IPython.display import clear_output
import hashlib
import uuid
import json, random
import argparse, sys, glob
import torch
from torch import nn
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import accelerate

import k_diffusion as K
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False, device='cuda'):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model = model.half().to(device)
    model.eval()
    return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def do_run(param_hash, accelerator, device, model, config, opt):
    # opt = SimpleNamespace(**opt)
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    metapath = f"{opt.outdir}/meta"
    
    os.makedirs(opt.outdir, exist_ok=True)
    os.makedirs(metapath, exist_ok=True)
    f = open(os.path.join(metapath, f"prompt.txt"), "w")
    f.write(opt.prompt)
    f.close()
    outpath = opt.outdir
    

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = outpath
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    with torch.no_grad():
        with model.ema_scope():
            with torch.cuda.amp.autocast():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling", disable=True):
                    for prompts in tqdm(data, desc="data", disable=True):
                        uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H//opt.f, opt.W//opt.f]
                        sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                        samples_ddim = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=True)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process and not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                print(param_hash)
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{param_hash}.png"))
                                f = open(os.path.join(metapath, f"{param_hash}.json"), "w")
                                f.write(p)
                                f.close()
                                base_count += 1
                        all_samples.append(x_samples_ddim)
                toc = time.time()

    return img

# if not model:
accelerator = accelerate.Accelerator()
device = accelerator.device
config = OmegaConf.load(f"{config}")
model = load_model_from_config(config, f"{ckpt}", device=device)

def get_database():
    from pymongo import MongoClient
    import pymongo

    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    from pymongo import MongoClient

    client = MongoClient(MONGODB_CONNECTION)
    return client

linode_obj_config = {
    "aws_access_key_id": AWS_ACCESS_KEY_ID,
    "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    "endpoint_url": ENDPOINT_URL,
}
s3_client = boto3.client("s3", **linode_obj_config)
# s3_client.create_bucket(Bucket='stockphotos')

with get_database() as client:
    operations = []
    records = list(client.stockphotos.stockphotos80.aggregate([
        {
            "$match" : {
                "$or" : [
                    {
                        "render_metadata" : {"$exists" : False}
                    },
                    {
                        "render_metadata.worker" : WORKER,
                        "render_metadata.rendered" : False 
                    }
                ],
                "image_type" : {"$in" : ["photo","illustration"]},
                "image_scores_popularity_label" : "HIGH"
            }
        },{
            "$limit" : BATCH_SIZE
        }
    ]))
    # Reserve Records
    for record in records:
        operations.append(
            UpdateOne({'_id': record.get('_id')}, {'$set': {'render_metadata': {
                "worker" : WORKER,
                "rendered" : False
            }}})
        )
    
    result = client.stockphotos.stockphotos80.bulk_write(operations)
       
    if records:
        for record in records:
            seed = 0
            id = record.get("_id")
            # Prompt
            prompt = record.get("description")
            
            # Image Type
            image_type = record.get("image_type")
            if image_type == "illustration":
                prompt = f"An illustration of {prompt}"
            
            # Width/Height
            w = 512
            h = 512
            aspect = record.get("aspect")
            if aspect:
                if aspect < 1:
                    w = 512
                    h = 704
                if aspect > 1:
                    w = 704
                    h = 512
                    
            batch = "graham/out"
            for offset in list(range(1)):
                u = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
                # For render
                opt = {
                    "prompt" : prompt,
                    "outdir" : f"/out/{batch}/{u}",
                    "skip_grid" : False,
                    "skip_save" : False,
                    "ddim_steps" : 50,
                    "plms" : False,
                    "ddim_eta" : 0.0,
                    "n_iter" : 1,
                    "W" : w,
                    "H" : h,
                    "C" : 4,
                    "f" : 8,
                    "n_samples" : 1,
                    "n_rows" : 2,
                    "scale" : 8,
                    "dyn" : None,
                    "from_file": None,
                    "seed" : seed + offset
                }
                # For hash
                params = {
                    "prompt" : prompt,
                    "skip_grid" : False,
                    "skip_save" : False,
                    "ddim_steps" : 50,
                    "plms" : False,
                    "ddim_eta" : 0.0,
                    "n_iter" : 1,
                    "W" : w,
                    "H" : h,
                    "C" : 4,
                    "f" : 8,
                    "n_samples" : 1,
                    "n_rows" : 2,
                    "scale" : 8,
                    "dyn" : None,
                    "from_file": None,
                    "seed" : seed + offset
                }
                opt = SimpleNamespace(**opt)
                p = json.dumps(params, indent = 4)
                param_hash = hashlib.sha256(p.encode('utf-8')).hexdigest()
                print(f"{prompt} | Seed: {seed} | [{w}x{h}] = {param_hash}")
                
                filename = os.path.join(opt.outdir, f"{param_hash}.png")
                
                if not os.path.exists(filename):
                    do_run(param_hash, accelerator, device, model, config, opt)
                
                # display(Img(filename=filename))
                
            s3_client.upload_file(
                Filename = filename,
                Bucket = 'stockphotos',
                Key = f"{param_hash}.png",
                ExtraArgs = {
                    "ACL" : "public-read",
                    "ContentType": "image/png"
                }
            )
            client.stockphotos.stockphotos80.update_one({"_id" : id}, {"$set" : {
                    "render_metadata.rendered" : True,
                    "render_metadata.hash" : param_hash,
                    "render_metadata.params" : params,
                }
            })
