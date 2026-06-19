import subprocess
import json
process = subprocess.Popen(['python3', './stdio/server.py'], 
                           stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE,
                           text=True)

list_tools_message = {"jsonrpc" : "2.0",
                      "id" : 1,
                      "method" : "tools/list",
                      "params" : {}
                      }

message = 'hello\n'

def send_message(message):
    print(f'[CLIENT] Sending message to server... Message: {message.strip()}')
    process.stdin.write(message)
    process.stdin.flush()

def serialise_message(message):
    return json.dumps(message) + '\n'

send_message(message)

response = process.stdout.readline()
print('[SERVER]:', response.strip())

send_message('exit')

process.stdin.close()
exit_code = process.wait()
print(f"Child exited with code {exit_code}")