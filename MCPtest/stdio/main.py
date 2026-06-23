from stdio.client import serialise_message, send_message
from stdio.utils.messages import list_tools_message, initialise_request, initialise_response, initialised_message
import subprocess
import json

process = subprocess.Popen(['python3', './stdio/server.py'], 
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        text=True)



def connect():
    print('Connecting to server...')
    
    #Ask server for capabilities
    send_message(serialise_message(initialise_request))

    #Read response from server
    response = process.stdout.readline()
    print('[SERVER]:', response.strip())

    #Send initialisation notification (handshake done)
    send_message(serialise_message(initialised_message))

def list_tools():
    #Send message to list tools
    send_message(serialise_message(list_tools_message))

    response = process.stdout.readline()
    return json.loads(response)['result']['tools']

def call_tool(tool_name, args):
    tool_message = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "args": args
        },
        "id": 1
    }

    send_message(serialise_message(tool_message))

    response = process.stdout.readline()
    return json.loads(response)['result']['properties']['content']['items']


def close_server():
    send_message('exit\n')
    exit_code = process.wait()
    print('Server exited with code:', exit_code)


def main():
    tools = []
    connect()
    tool_response = list_tools()
    tools.extend(tool_response)

    print('Available tools:', tools)

    tool = tools[0]
    tool_call_response = call_tool(tool['name'], {"arg1": "hello"})
    for content in tool_call_response:
        print(f'[SERVER]: tool response:{content["text"]}\n')

    close_server()

main()





print("hello")