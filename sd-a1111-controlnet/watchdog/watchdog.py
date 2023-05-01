import requests, os, time, traceback, json
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime, timedelta
import xmlrpc.client

def loop(args):
    # Start bot loop
    run = True
    idle_time = 0
    boot_time = datetime.utcnow()
    # fd_api_url = f"{args['api']}/v3/agentstatus"
    urls = f"{args['api']}".split(",")
    poll_interval = 3

    while run:
        time.sleep(poll_interval)
        try:
            # Get A1111 Status
            progress = requests.get(
                "http://localhost:7860/sdapi/v1/progress?skip_current_image=true",
                headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            ).json()

            # logger.info(progress)

            memory = requests.get(
                "http://localhost:7860/sdapi/v1/memory",
                headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            ).json()

            # logger.info(memory)
            payload = {
                "agent_id" : args["agent"],
                "progress" : progress,
                "memory" : memory
            }
            # logger.info(f"Payload:\n{payload}")
            if payload["memory"]["cuda"]["events"]["oom"] >= 3:
                # Connect to the supervisord XML-RPC API
                server = xmlrpc.client.ServerProxy('http://localhost:9001/RPC2')
                logger.info("☠️ Restarting A1111, OOMs detected...")
                result = server.supervisor.stopProcess('auto1111')
                logger.info(result)
                result = server.supervisor.startProcess('auto1111')
                logger.info(result)

            for base_url in urls:
                url = f"{base_url}/v3/agentstatus"
                try:
                    fd_results = requests.post(
                        url,
                        headers = {'Content-type': 'application/json'},
                        data=json.dumps(payload)
                    ).json()
                except Exception as e:
                    logger.info(f"📡 Cannot reach FD API {url}")
                    tb = traceback.format_exc()
                    logger.error(tb)
                    pass
            # logger.info(f"🐶 Sleeping for {poll_interval} seconds...")

        except requests.exceptions.ConnectionError as e:
            # tb = traceback.format_exc()
            # logger.error(tb)
            logger.info("📡 Cannot reach A1111 API.  Maybe it is just starting or crashed?")
            time.sleep(10)
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
