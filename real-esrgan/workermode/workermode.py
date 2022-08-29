import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
load_dotenv()

def loop(args):
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
    
        gpu_record = json.dumps(gpu_record)
        memdict = json.dumps(memdict)
        
        url = f"{args.api}/v3/takeorder/{args.agent}"
        
        try:
            logger.debug(f"🌎 Checking '{url}'...")
            results = requests.post(
                url,
                data={
                    "bot_version": AGENTVERSION,
                    "algo" : "esrgan",
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
                    do_job(args, details)
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
    
    # determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name in ['RealESRGAN_x4plus', 'RealESRNet_x4plus']:  # x4 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x4plus_anime_6B']:  # x4 RRDBNet model with 6 blocks
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name in ['RealESRGAN_x2plus']:  # x2 RRDBNet model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name in ['realesr-animevideov3']:  # x4 VGG-style model (XS size)
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4

    # determine model paths
    model_path = os.path.join('/workspace/Real-ESRGAN/experiments/pretrained_models', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('/workspace/Real-ESRGAN/realesrgan/weights', args.model_name + '.pth')
    if not os.path.isfile(model_path):
        raise ValueError(f'Model {args.model_name} does not exist.')

    # restorer
    print("Loading restorer")
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    print("Loading face enhancer.")
    from gfpgan import GFPGANer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler)

    os.makedirs(args.output, exist_ok=True)

    job = {
        image : "https://images.feverdreams.app/images/e8890f5252bf20438e090210a1a2e2cf8940a32bb926cffe912a0dd00d2125a6.png"
    }
    
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        try:
            if args.face_enhance:
                _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if args.ext == 'auto':
                extension = extension[1:]
            else:
                extension = args.ext
            if img_mode == 'RGBA':  # RGBA images should be saved in png format
                extension = 'png'
            if args.suffix == '':
                save_path = os.path.join(args.output, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(args.output, f'{imgname}_{args.suffix}.{extension}')
            cv2.imwrite(save_path, output)

def main():
    """Inference demo for Real-ESRGAN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", help="Worker API http endpoint", required=True, default=os.getenv("DD_API"))
    parser.add_argument("--agent", help="Worker agent name", required=True, default=os.getenv("AGENTNAME"))
    parser.add_argument("--owner", help="Worker agent owner", required=False, default=398901736649261056)
    parser.add_argument("--images_out", help="Directory for render jobs", required=False, default=os.getenv("DD_IMAGES_OUT", "/workspace/out"))
    parser.add_argument("--poll_interval", type=int, help="Polling interval between jobs", required=False, default=os.getenv("DD_POLLINTERVAL", 0))
    parser.add_argument("--dream_time", type=int, help="Time in seconds until dreams", required=False, default=os.getenv("DD_POLLINTERVAL", 300))   
    
    # parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    # parser.add_argument(
    #     '-n',
    #     '--model_name',
    #     type=str,
    #     default='RealESRGAN_x4plus',
    #     help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
    #           'realesr-animevideov3'))
    # parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    # parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    # parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    # parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    # parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    # parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    # parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    # parser.add_argument(
    #     '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    # parser.add_argument(
    #     '--alpha_upsampler',
    #     type=str,
    #     default='realesrgan',
    #     help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    # parser.add_argument(
    #     '--ext',
    #     type=str,
    #     default='auto',
    #     help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    # parser.add_argument(
    #     '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    args = parser.parse_args()
    loop(args)

if __name__ == '__main__':
    main()