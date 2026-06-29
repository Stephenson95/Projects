import random
import threading

def create_sampling_message(product):
    sampling_message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "sampling/createMessage",
        "params": {
            "messages": [{
                "role": "system",
                "content": {
                    "type" : "text",
                    "text" : f"New product available: {product['name']} (ID: {product['id']}), Price: ${product['price']}. Keywords: {', '.join(product['keywords'])}"
                }
            }],
            "systemPrompt": "You are a helpful assistant assisting with product descriptions",
            "includeContext": "thisServer",
            "maxTokens": 300
            }
        }
    return sampling_message


class ProductStore:
    def __init__(self):
        self.started = False
        self.listeners = {}
        #Create timer that adds a product every 5 seconds
    
    def add_product(self):
        product = {
            "id": str(random.randint(10000, 99999)),
            "name": f"Product {random.randint(1, 100)}",
            "price": round(random.uniform(10, 100), 2),
            "keywords": [f"keyword{random.randint(1, 5)}" for _ in range(random.randint(1,3))]
        }
        self.dispatch_message("new_product", product)
    
    def start_product_queue_timer(self):
        """Start a timer that adds a product every 5 seconds"""
        def schedule_next():
            delay = random.uniform(1,2)
            self.product_timer = threading.Timer(delay, self.add_product)
            self.product_timer.start()
        
        def add_twice():
            schedule_next()
            schedule_next()
        
        add_twice()

    def add_listener(self, message, callback):
        if not self.started:
            self.started = True
            self.start_product_queue_timer()
        callbacks = self.listeners.get(message, [])
        callbacks.append(callback)
        self.listeners[message] = callbacks


    def dispatch_message(self, message, payload):
        callbacks = self.listeners.get(message, [])
        for callback in callbacks:
            callback(payload)
