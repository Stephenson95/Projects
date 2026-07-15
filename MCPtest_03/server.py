from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uuid

mcp = FastMCP("Demo")

class Customer(BaseModel):
    id: int
    name: str
    email: str

    def __init__(self, **data):
        super().__init__(**data)

class Category(BaseModel):
    id: int
    name: str
    description: str

    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = uuid.uuid4().int
        super().__init__(**data)

class Product(BaseModel):
    id: int
    name: str
    price: float
    description: str

class CartItem(BaseModel):
    cart_id: uuid.UUID
    product_id: int
    quantity: int

    def __init__(self, cart_id: uuid.UUID, product_id: int, quantity: int):
        if cart_id != uuid.UUID(int=0):
            self.cart_id = cart_id
        else:
            self.cart_id = uuid.uuid4()

        self.product_id = product_id
        self.quantity = quantity


class Cart(BaseModel):
    id: int
    customer_id: int
    
    def __init__(self, **data):
        if 'id' not in data:
            data['id'] = uuid.uuid4().int
        super().__init__(**data)
    
class Order(BaseModel):
    id: int
    customer_id: int

    def __init__(self, **data):
        if 'id' not in data or not isinstance(data['id'], uuid.UUID):
            data['id'] = uuid.uuid4().int
        super().__init__(**data)

products = [
    Product(id=1, name="Product 1", price=10.0, description="Description of product 1"),
    Product(id=2, name="Product 2", price=20.0, description="Description of product 2"),
    Product(id=3, name="Product 3", price=30.0, description="Description of product 3"),
]

orders = [
    Order(id=1, customer_id=101),
    Order(id=uuid.uuid4().int, customer_id=101),
    Order(id=uuid.uuid4().int, customer_id=102)
]

carts = []
cart_items = []

customers = [
    Customer(id=101, name='Steven Li', email='steven.li@example.com'),
]

categories = [
    Category(id=uuid.uuid4().int, name='Electronics', description='Electronic devices and gadgets'),
    Category(id=uuid.uuid4().int, name='Clothing', description='Apparel and accessories'),
    Category(id=uuid.uuid4().int, name='Books', description='Books and literature')
]

product_category = [
    {
        "name": "Product 1",
        "price": 10.0,
        "description": "Description of product 1",
        "category_id": 1
    },
    {
        "name": "Product 2",
        "price": 20.0,
        "description": "Description of product 2",
        "category_id": 2
    },
    {
        "name": "Product 3",
        "price": 30.0,
        "description": "Description of product 3",
        "category_id": 3
    }
]


#Retrive orders for customer id
@mcp.tool()
def get_orders(customer_id:int = 0) -> list:
    """Return list of orders"""
    if customer_id != 0 and not any(customer_id == customer.id for customer in customers):
        raise ValueError(f"Invalid customer id: {customer_id}")
    
    filtered_orders = orders
    if customer_id != 0:
        filtered_orders = [order for order in orders if order.customer_id == customer_id]
    
    return filtered_orders
    
@mcp.tool()
def get_order(order_id:int) -> Order|None:
    """Return order"""
    for order in orders:
        if order.id == order_id:
            return order

    return None


@mcp.tool()
def place_order(customer_id:int) -> Order:
    """Place an order for the specified customer"""
    if customer_id != 0 and not any(customer_id == customer.id for customer in customers):
        raise ValueError(f"Invalid customer id: {customer_id}")
    
    new_order = Order(id=uuid.uuid4().int, customer_id=customer_id)
    orders.append(new_order)
    return new_order

@mcp.tool()
def get_cart(customer_id:int) -> Cart|None:
    """Get cart for the specified customer"""
    if customer_id != 0 and not any(customer_id == customer.id for customer in customers):
        raise ValueError(f"Invalid customer id: {customer_id}")

    cart = next((cart for cart in carts if cart.customer_id == customer_id), None)

    return cart


@mcp.tool()
def get_cart_items(cart_id:int) -> List[Cart]:
    """Get cart items for the specified cart id"""
    if cart_id != 0 and not any(cart_id == cart.id for cart in carts):
        raise ValueError(f"Invalid cart id: {cart_id}")

    items = [item for item in cart_items if item.cart_id == cart_id]
    return items


@mcp.tool()
def add_to_cart(cart_id:int, product_id:int, quantity:int) -> CartItem:
    """Add an item to the specified cart"""
    new_cart_item = CartItem(cart_id=cart_id, product_id=product_id, quantity=quantity)
    cart_items.append(new_cart_item)
    return new_cart_item

@mcp.tool()
def get_all_products() -> List[Product]:
    """Return list of all products"""
    return products

@mcp.tool()
def get_product(product_id:int) -> Product|None:
    """Return a specific product by ID"""
    for product in products:
        if product.id == product_id:
            return product
    return None

@mcp.tool()
def get_all_categories() -> List[Category]:
    """Return list of all categories"""
    return categories

@mcp.tool()
def get_customers() -> List[Customer]:
    """Return list of all customers"""
    return customers


@mcp.resource("resource:product_category")
def product_catalog() -> list:
    """Returns product category"""
    return product_category