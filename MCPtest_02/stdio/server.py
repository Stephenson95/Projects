import sys
import json
from stdio.utils.messages import initialise_response, progress_notification
from stdio.external_server import ProductStore, create_sampling_message

def send_response(response):
    print(json.dumps(response))
    sys.stdout.flush()

def handle_sampling_response(response):
    content = response['result']['content']['text']
    print("[SERVER] [Sampling response received]:", content)
    sys.stdout.flush()
    #TODO, update the store or perform any other action with the response

initialised = False

product_sample = {
    "id": "12345",
    "name": "Sample Product",
    "price": 19.99,
    "keywords": ["sample", "product", "example"]
}

store = ProductStore()
store.add_listener("new_product", lambda product: print(json.dumps(create_sampling_message(product)) and sys.stdout.flush()))

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
                    #Prepare tool list response
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
                    #Send notification to say we're working on it
                    send_response(create_sampling_message(product_sample))
                    send_response(progress_notification)

                    #Extract tool name and arguments
                    tool_name = json_message['params']['name']
                    tool_args = json_message['params']['args']

                    #Prepare response for tool call
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
                    if json_message['result']:
                        handle_sampling_response(json_message)
                    else:
                        send_response(f"Unknown method: {method}")
                    break

        elif message == "exit":
            send_response("Exiting server.")
            sys.exit(0)
        else:
            send_response(f"Unrecognised message: {message}")