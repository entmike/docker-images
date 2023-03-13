import requests, os, time, traceback
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime, timedelta

def loop(args):
    # Start bot loop
    run = True
    idle_time = 0
    boot_time = datetime.utcnow()
    fd_api_url = f"{args['api']}/v3/agentstatus"

    while run:
        try:
            # Get A1111 Status
            progress = requests.get(
                "http://localhost:7860/sdapi/v1/progress?skip_current_image=true",
                headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            ).json()

            # logger.info(progress)

            memory = requests.get(
                "http://localhost:7860/sdapi/v1/memory",
                headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
            ).json()

            # logger.info(memory)
            
            try:
                fd_results = requests.post(
                    fd_api_url,
                    data={
                        "progress" : progress,
                        "memory" : memory
                    }
                ).json()

            except Exception as e:
                logger.info("📡 Cannot reach FD API.")
                tb = traceback.format_exc()
                logger.error(tb)
                pass

            
            poll_interval = 3
            logger.info(f"🐶 Sleeping for {poll_interval} seconds...")
            time.sleep(poll_interval)

        except requests.exceptions.ConnectionError as e:
            # tb = traceback.format_exc()
            # logger.error(tb)
            logger.info("📡 Cannot reach A1111 API.  Maybe it is just starting or crashed?")
            pass
        
        except:
            tb = traceback.format_exc()
            logger.error(f"🐶❌\n{tb}")
            
if __name__ == "__main__":
    # Environment-specific settings:
    load_dotenv()
    cliargs = {
        "api" : os.environ.get('API_URL', ""),
        "agent" : os.environ.get('API_AGENT', ""),
        "owner" : os.environ.get('API_OWNER', ""),
        "out" : "/outdir",
        "poll_interval" : 5
    }
    
    # Start bot loop
    logger.info("➰ Starting watchdog loop...")
    loop(cliargs)
