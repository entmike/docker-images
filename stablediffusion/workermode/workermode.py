import shutil
import nvsmi
import subprocess
from loguru import logger
import time, requests, json, os
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')
import argparse, os, sys, glob
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

AGENTVERSION = "sd-1-4-v1.2"

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

def do_run(accelerator, device, model, config, opt):
    from types import SimpleNamespace
    opt = SimpleNamespace(**opt)
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    os.makedirs(opt.outdir, exist_ok=True)
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

    sample_path = os.path.join(outpath, opt.uuid)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    with torch.no_grad():
        with model.ema_scope():
            with torch.cuda.amp.autocast():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling", disable=not accelerator.is_main_process):
                    for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                        uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H//opt.f, opt.W//opt.f]
                        sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {'cond': c, 'uncond': uc, 'cond_scale': opt.scale}
                        
                        sampler = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        
                        try:
                            sampler_method = opt.sampler
                        except:
                            sampler_method = "k_lms"

                        if sampler_method == "ddim":
                            sampler = DDIMSampler(model_wrap_cfg)
                        if sampler_method == "plms":
                            sampler = PLMSSampler(model_wrap_cfg)
                        if sampler_method == "k_lms":
                            sampler = K.sampling.sample_lms(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        if sampler_method == "k_euler":
                            sampler = K.sampling.sample_euler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        if sampler_method == "k_euler_ancestral":
                            sampler = K.sampling.sample_euler_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        if sampler_method == "k_heun":
                            sampler = K.sampling.sample_heunl(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        if sampler_method == "k_dpm_2":
                            sampler = K.sampling.sample_dpm_2(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        if sampler_method == "k_dpm_2_ancestral":
                            sampler = K.sampling.sample_dpm_2_ancestral(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)

                        x_samples_ddim = model.decode_first_stage(sampler)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        x_samples_ddim = accelerator.gather(x_samples_ddim)

                        if accelerator.is_main_process and not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1
                        all_samples.append(x_samples_ddim)

                if accelerator.is_main_process and not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

    if accelerator.is_main_process:
        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f"Sampling took {toc-tic:g}s, i.e. produced {opt.n_iter * opt.n_samples * accelerator.num_processes / (toc - tic):.2f} samples/sec.")
    
def do_job(args, details, accelerator, device, model, config):
    import os, time
    params = details["params"]
    # Grab start timestamp
    start_time = time.time()
    try:
        sampler = params["sampler"]
    except:
        sampler = "k_lms"

    # Run Stable
    try:
        opt = {
            "uuid" : details["uuid"],
            
            # User params
            "ddim_steps" : params["steps"],
            "ddim_eta" : params["eta"],
            "W" : params["width_height"][0],
            "H" : params["width_height"][1],
            "seed": params["seed"],
            "prompt" : params["prompt"],
            "scale" : params["scale"],
            "sampler" : sampler,
            "outdir" : args.out,
            "skip_grid" : True,
            "skip_save" : False,
            "plms" : False,
            "n_iter" : 1,
            "C" : 4,
            "f" : 8,
            "n_samples" : 1,
            "n_rows" : 2,
            "dyn" : None,
            "from_file": None,
        }
        results = do_run(accelerator, device, model, config, opt)

        # Grab end timestamp
        end_time = time.time()

        # Capture duration
        duration = end_time - start_time
        deliver(args, details, duration)
    except Exception as e:
        connected = True #TODO: what?
        if connected:
            tb = traceback.format_exc()
            logger.error(f"Bad job detected.\n\n{e}\n\n{tb}")
            values = {"message": f"Job failed:\n\n{e}", "traceback": tb}
            requests.post(f"{args.api}/v3/reject/{args.agent}/{details['uuid']}", data=values)
        else:
            logger.error(f"Error.  Check your API host is running at that location.  Also check your own internet connectivity.  Exception:\n{tb}")
            raise(tb)

def deliver(args, details, duration):
    sample_path = os.path.join(args.out, details["uuid"])
    image = f"{sample_path}/00000.png"
    url = f"{args.api}/v3/deliverorder"
    logger.info(url)
    
    files = {
        "file": open(image, "rb")
    }
    values = {
        "duration" : duration,
        "agent_id" : args.agent,
        "algo" : "stable",
        "agent_version" : AGENTVERSION,
        "owner" : args.owner,
        "uuid" : details['uuid']
    }
    # Upload payload
    try:
        logger.info(f"🌍 Uploading {image} to {url}...")
        results = requests.post(url, files=files, data=values)
        try:
            feedback = results.json()
        except:
            feedback = results.text
        logger.info(feedback)
    except:
        logger.error("Error uploading image.")
        import traceback
        tb = traceback.format_exc()
        logger.error(tb)

    if not os.getenv('WORKER_SAVEFILES'):
        try:
            # Delete sample
            os.unlink(image)
            # Clean up directory
            shutil.rmtree(sample_path) 
        except:
            logger.error(f"Error when trying to clean up files for {details['uuid']}")
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)

def loop(args, accelerator, device, config, model):
    # Start bot loop
    run = True
    idle_time = 0
    boot_time = datetime.utcnow()
    
    while run:
        start_time = time.time()
        gpu = list(nvsmi.get_gpus())[0]
        total, used, free = shutil.disk_usage(args.out)
        import psutil
        try:
            meminfo = psutil.virtual_memory()
            memdict = {}
            memdict['total'] = meminfo.total/1024
            memdict['used'] = meminfo.used/1024
            memdict['free'] = meminfo.free/1024
            memdict['buffers'] = meminfo.buffers/1024
            memdict['cached'] = meminfo.cached/1024
            memdict['percent'] = meminfo.percent
        except:
            memdict = {}

        gpu_record = {}
        for key in list(gpu.__dict__.keys()):
            gpu_record[key]=gpu.__dict__[key]
        
        vram = gpu_record["mem_total"]
        if vram > 50000:
            text_clip_on_cpu = True
        else:
            text_clip_on_cpu = False

        gpu_record = json.dumps(gpu_record)
        memdict = json.dumps(memdict)
        
        url = f"{args.api}/v3/takeorder/{args.agent}"
        try:
            logger.debug(f"🌎 Checking '{url}'...")
            results = requests.post(
                url,
                data={
                    "bot_version": AGENTVERSION,
                    "algo" : "stable",
                    "gpus": gpu_record,
                    "owner": args.owner,
                    "idle_time": idle_time,
                    "start_time" : start_time,
                    "free_space" : free,
                    "total_space" : total,
                    "used_space" : used,
                    "boot_time" : boot_time,
                    "memory" : memdict
                }
            ).json()
            
            if "command" in results:
                if results["command"] == 'terminate':
                    logger.info("🛑 Received terminate instruction.  Cya.")
                    run = False    
                    
            if results["success"]:
                if "details" in results:
                    details = results["details"]
                    logger.info(f"Job {details['uuid']} received.")
                    idle_time = 0
                    do_job(args, details, accelerator, device, model, config)
        
            else:
                logger.error(results)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(tb)
            pass
        
        if run:
            poll_interval = args.poll_interval
            poll_interval = 5
            logger.info(f"Sleeping for {poll_interval} seconds...  I've been sleeping for {idle_time} seconds.")
            time.sleep(poll_interval)
            idle_time = idle_time + poll_interval
        else:
            logger.info("Terminating loop.")

def main():
    # Environment-specific settings:
    load_dotenv()

    import argparse
    parser = argparse.ArgumentParser(description="Agent Worker")
    parser.add_argument("--api", help="Worker API http endpoint", required=True, default=os.getenv("DD_API"))
    parser.add_argument("--agent", help="Worker agent name", required=True, default=os.getenv("AGENTNAME"))
    parser.add_argument("--owner", help="Worker agent owner", required=False, default=398901736649261056)
    parser.add_argument("--images_out", help="Directory for render jobs", required=False, default=os.getenv("DD_IMAGES_OUT", "/workspace/out"))
    parser.add_argument("--poll_interval", type=int, help="Polling interval between jobs", required=False, default=os.getenv("DD_POLLINTERVAL", 0))
    parser.add_argument("--dream_time", type=int, help="Time in seconds until dreams", required=False, default=os.getenv("DD_POLLINTERVAL", 300))
    parser.add_argument("--ckpt", help="Path to checkpoint", required=False, default=os.getenv("CKPT", "/weights/sd-v1-3-full-ema.ckpt"))
    parser.add_argument("--config", help="Path to diffusion config", required=False, default=os.getenv("CONFIG", "/workspace/k-diffusion/v1-inference.yaml"))
    parser.add_argument("--out", help="Path to output", required=False, default=os.getenv("CONFIG", "/workspace/out"))
    cliargs = parser.parse_args()
    
    # Load model first
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    config = OmegaConf.load(cliargs.config)
    model = load_model_from_config(config, cliargs.ckpt, device=device)
    
    loop(cliargs, accelerator, device, config, model)

if __name__ == "__main__":
    main()
