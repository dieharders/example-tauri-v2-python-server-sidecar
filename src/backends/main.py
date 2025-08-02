import os
import signal
import sys
import asyncio
import threading
from typing import TypedDict
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from inference import infer_text_api
from uvicorn import Config, Server

PORT_API = 8008

server_instance = None  # Global reference to the Uvicorn server instance

app = FastAPI(
    title="API server",
    version="0.1.0",
)

# Configure CORS settings
origins = [
    "*",  # to whitelist any url, REMOVE THIS FOR PRODUCTION!!!
    # "http://localhost:3000", # for dev
    # "https://your-web-ui.com", # for prod
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Tell client we are ready to accept requests.
# This is a mock func, modify to your needs.
@app.get("/v1/connect")
def connect_to_api_server():
    print("[server] Connecting to server...", flush=True)
    host = f"http://localhost:{PORT_API}"
    return {
        "message": f"Connected to api server on port {PORT_API}. Refer to '{host}/docs' for api docs.",
        "data": {
            "port": PORT_API,
            "pid": os.getpid(),
            "host": host,
        },
    }


class T_Query(TypedDict):
    prompt: str


# Mock text inference endpoint, here for inspiration.
@app.post("/v1/completions")
def llm_completion(payload: T_Query = Body(...)):
    return infer_text_api.completions(payload)


# Programmatically force shutdown this sidecar.
def kill_process():
    os.kill(os.getpid(), signal.SIGINT)  # This force closes this script.


# Graceful shutdown handler
async def shutdown_server():
    global server_instance
    if server_instance:
        print("[sidecar] Shutting down server gracefully...", flush=True)
        await server_instance.shutdown()


# Programmatically startup the api server
async def start_api_server_async(**kwargs):
    global server_instance
    port = kwargs.get("port", PORT_API)
    try:
        if server_instance is None:
            print("[sidecar] Starting API server...", flush=True)
            config = Config(app, host="0.0.0.0", port=port, log_level="info")
            server_instance = Server(config)
            # Start the ASGI server
            await server_instance.serve()
        else:
            print(
                "[sidecar] Failed to start new server. Server instance already running.",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[sidecar] Received shutdown signal, stopping server...", flush=True)
        await shutdown_server()
    except Exception as e:
        print(f"[sidecar] Error, failed to start API server {e}", flush=True)


def start_api_server(**kwargs):
    try:
        asyncio.run(start_api_server_async(**kwargs))
    except KeyboardInterrupt:
        print("[sidecar] Server shutdown complete.", flush=True)
    except Exception as e:
        print(f"[sidecar] Unexpected error during server operation: {e}", flush=True)


# Handle the stdin event loop. This can be used like a CLI.
def stdin_loop():
    print("[sidecar] Waiting for commands...", flush=True)
    while True:
        try:
            # Read input from stdin.
            user_input = sys.stdin.readline().strip()

            # If stdin is closed (EOF), break the loop
            if not user_input and sys.stdin.closed:
                break

            # Check if the input matches one of the available functions
            match user_input:
                case "sidecar shutdown":
                    print("[sidecar] Received 'sidecar shutdown' command.", flush=True)
                    kill_process()
                case _:
                    if user_input:  # Only print if there's actual input
                        print(
                            f"[sidecar] Invalid command [{user_input}]. Try again.",
                            flush=True,
                        )
        except (EOFError, KeyboardInterrupt):
            print("[sidecar] Input handler shutting down...", flush=True)
            break
        except Exception as e:
            print(f"[sidecar] Error in input handler: {e}", flush=True)


# Start the input loop in a separate thread
def start_input_thread():
    try:
        input_thread = threading.Thread(target=stdin_loop)
        input_thread.daemon = True  # so it exits when the main program exits
        input_thread.start()
    except Exception as e:
        print(f"[sidecar] Failed to start input handler: {e}", flush=True)


# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    print(f"[sidecar] Received signal {signum}, initiating shutdown...", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # You can spawn sub-processes here before the main process.
    # new_command = ["python", "-m", "some_script", "--arg", "argValue"]
    # subprocess.Popen(new_command)

    # Listen for stdin from parent process
    start_input_thread()

    # Starts API server, blocks further code from execution.
    start_api_server()
