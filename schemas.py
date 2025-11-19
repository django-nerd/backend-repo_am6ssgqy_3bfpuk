"""
Database Schemas for Restaurant Ordering App

Each Pydantic model maps to a MongoDB collection (lowercased class name)
- User -> user
- Restaurant -> restaurant
- MenuItem -> menuitem
- Order -> order
- OrderEvent -> orderevent (optional stream of updates)
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Literal
from datetime import datetime

class User(BaseModel):
    """End-user or restaurant staff account
    role: 'customer' or 'restaurant'
    oauth_provider/id are recorded after OAuth login
    """
    name: str = Field(..., description="Full name")
    email: EmailStr = Field(..., description="Email address")
    role: Literal['customer', 'restaurant'] = Field('customer')
    oauth_provider: Optional[str] = Field(None, description="e.g. google, github")
    oauth_id: Optional[str] = Field(None, description="Provider user id")
    avatar_url: Optional[str] = None
    is_active: bool = True

class Restaurant(BaseModel):
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None
    owner_user_id: Optional[str] = Field(None, description="Links to user._id (restaurant role)")

class MenuItem(BaseModel):
    restaurant_id: str
    name: str
    description: Optional[str] = None
    price_cents: int = Field(..., ge=0)
    is_available: bool = True
    image_url: Optional[str] = None

class OrderItem(BaseModel):
    item_id: str
    name: str
    quantity: int = Field(1, ge=1)
    unit_price_cents: int = Field(..., ge=0)

class Order(BaseModel):
    restaurant_id: str
    customer_id: str
    items: List[OrderItem]
    status: Literal['pending','accepted','preparing','ready','completed','cancelled'] = 'pending'
    note: Optional[str] = None
    total_cents: int = Field(..., ge=0)
    placed_at: Optional[datetime] = None

class OrderEvent(BaseModel):
    order_id: str
    type: Literal['status_changed','new_order','cancelled']
    status: Optional[str] = None
    message: Optional[str] = None
