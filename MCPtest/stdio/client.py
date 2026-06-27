from stdio.utils.messages import list_tools_message, initialise_request, initialise_response, initialised_message
import subprocess
import json
import sys
import queue
import threading

message_queue = queue.Queue()

process = subprocess.Popen([sys.executable, '-m', 'stdio.server'], 
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           text=True)


def is_sampling_message(message):
    return message.get('method', '').startswith('sampling')

def is_notification_message(message):
    return message.get('method', '').startswith('notifications/')

def create_sampling_message(llm_response):
    sampling_message = {
        "jsonrpc": "2.0",
        "result": {
            "content": {
                "text": llm_response
                }
            }}
    return sampling_message

def call_llm(message):
    return "LLM: " + message

def handle_sampling_message(message):
    print("[CLIENT] Calling LLM to complete request", message)
    
    content = message['params']['messages'][0]['content']['text']
    llm_response = call_llm(content)
    message = create_sampling_message(llm_response)
    send_message(serialise_message(message))

def listen_to_stdout():
    while True:
        response = process.stdout.readline()
        if not response:
            break
        try:
            parsed_response = json.loads(response)
            if is_sampling_message(parsed_response):
                handle_sampling_message(parsed_response)
            else:
                message_queue.put(response.strip()) #if message is not a sampling message, put in queue
        except json.JSONDecodeError:
            print("[THREAD] Non-JSON response received:", response.strip())


def send_message(message):
    
    try:
        parsed = json.loads(message)
        print(f'[CLIENT] Sending message to server... Message: {json.dumps(parsed, indent=2)}')
    except json.JSONDecodeError:
        print(f'[CLIENT] Sending message to server... Message: {message.strip()}')
    process.stdin.write(message)
    process.stdin.flush()

def serialise_message(message):
    return json.dumps(message) + '\n'

def connect():
    print('Connecting to server...')
    
    #Ask server for capabilities
    send_message(serialise_message(initialise_request))

    #Read response from server
    #response = process.stdout.readline()
    response = message_queue.get()
    print('[SERVER]:', response.strip())

    #Send initialisation notification (handshake done)
    send_message(serialise_message(initialised_message))

def list_tools():
    #Send message to list tools
    send_message(serialise_message(list_tools_message))

    has_result = False
    while not has_result:
        #response = process.stdout.readline()
        response = message_queue.get()
        parsed_response = json.loads(response)
        #check if message has result attribute - if so break out
        if 'result' in parsed_response:
            has_result = True
            return parsed_response['result']['tools']
        else:
            print(f'[SERVER] notification: \n{response.strip()}')

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

    has_result = False
    send_message(serialise_message(tool_message))

    while not has_result:
        #response = process.stdout.readline()
        response = message_queue.get()
        #check if message has result attribute - if so break out
        parsed_response = json.loads(response)
        if 'result' in parsed_response:
            has_result = True
            return parsed_response['result']['properties']['content']['items']
        else:
            print(f'[SERVER] notification: \n{response.strip()}')

def close_server():
    #send_message('exit\n')
    exit_code = process.wait()
    print('Server exited with code:', exit_code)


listener_thread = threading.Thread(target=listen_to_stdout, daemon=True)
listener_thread.start()

def main():
    tools = []
    connect()
    tool_response = list_tools()
    tools.extend(tool_response)

    print('Available tools:', tools)

    tool = tools[0]
    tool_call_response = call_tool(tool['name'], {"arg1": "hello"})
    for content in tool_call_response:
        print(f'[SERVER]: tool response: {content["text"]}\n')

    close_server()

main()
