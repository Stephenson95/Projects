import sys
import json
from stdio.utils.messages import initialise_response

def send_response(response):
    print(json.dumps(response))
    sys.stdout.flush()

initialised = False

while True:
    for line in sys.stdin:
        message = line.strip()
        if message == "hello":
            send_response("hello matey!")
        elif message.startswith('{"jsonrpc":'):
            json_message = json.loads(message)
            method = json_message.get('method', '')

            if not initialised:
                if method != "initialise" and method != "notifications/initialised":
                    print(f"Server not initialised. Please send an initialised notification first. You sent {method}")
                    sys.stdout.flush()
                    continue

            match method:
                case "notifications/initialised":
                    sys.stdout.flush()
                    initialised = True
                    break
                case "initialise":
                    send_response(initialise_response)
                    break

                case "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": json_message["id"],
                        "result": {"tools": [
                                    {"name": "example_tool",
                                     "description": "This is a tool that does something",
                                     "inputSchema": {
                                        "type": "object",
                                        "properties": {
                                            "arg1": {
                                                    "type": "string",
                                                    "description": "This is an example argument"
                                            }
                                        }
                                     },
                                     "required": ["arg1"]
                                    }]}
                    }
                    send_response(response)
                    break


                case "tools/call":
                    tool_name = json_message['params']['name']
                    tool_args = json_message['params']['args']

                    response = {
                        "jsonrpc": "2.0",
                        "id": json_message["id"],
                        "result": {"properties": {
                                        "content": {
                                            "description": "description of the content",
                                            "items": [{
                                                "type": "text",
                                                "text": f"Called tool {tool_name} with arguments {tool_args}"
                                            }]
                                        }
                                    }
                        }
                    }
                    send_response(response)
                    break

                case _:
                    send_response(f"Unknown method: {method}")
                    break
        elif message == "exit":
            send_response("Exiting server.")
            sys.exit(0)
        else:
            send_response(f"Unrecognised message: {message}")