import sys
import json

def send_response(response):
    print(json.dumps(response))
    sys.stdout.flush()

while True:
    for line in sys.stdin:
        message = line.strip()
        if message == "hello":
            send_response("hello matey!")
        elif message.startswith('{"jsonrpc":'):
            json_message = json.loads(message)
            match json_message['method']:
                case "tools/list":
                    response = {
                        "jsonrpc": "2.0",
                        "id": json_message["id"],
                        "result": ["tool1", "tool2"]
                    }
                    send_response(response)
                    break
                case _:
                    send_response(f"Unknown method: {json_message['method']}")
                    break
        elif message == "exit":
            send_response("Exiting server.")
            sys.exit(0)
        else:
            send_response(f"Unrecognised message: {message}")