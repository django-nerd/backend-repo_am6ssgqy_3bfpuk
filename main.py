import os
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, EmailStr
from jose import jwt

from database import db, create_document, get_documents
from schemas import User, Restaurant, MenuItem, Order, OrderItem, OrderEvent

# Optional OAuth via Authlib (enabled if env vars are set)
OAUTH_AVAILABLE = False
try:
    from authlib.integrations.starlette_client import OAuth
    from starlette.config import Config as StarletteConfig
    OAUTH_AVAILABLE = True
except Exception:
    OAUTH_AVAILABLE = False

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = "HS256"
TOKEN_EXPIRE_MIN = 60 * 24 * 14  # 14 days
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app = FastAPI(title="Restaurant Ordering API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- Helpers ----------------------
try:
    from bson import ObjectId
except Exception:  # pragma: no cover
    ObjectId = None

def to_object_id(id_str: str):
    if ObjectId is None:
        raise HTTPException(status_code=500, detail="bson not available")
    try:
        return ObjectId(id_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id format")


# ---------------------- Auth & JWT ----------------------
class DemoAuthBody(BaseModel):
    email: EmailStr
    name: str = Field(...)
    role: str = Field('customer')  # 'customer' or 'restaurant'
    avatar_url: Optional[str] = None


def create_jwt(payload: Dict[str, Any]) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(minutes=TOKEN_EXPIRE_MIN)
    to_encode = {"exp": exp, "iat": now, **payload}
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)


def decode_jwt(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")


bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Dict[str, Any]:
    if not creds:
        raise HTTPException(status_code=401, detail="Authorization required")
    data = decode_jwt(creds.credentials)
    return data


@app.post("/auth/demo")
def demo_auth(body: DemoAuthBody):
    """Simple, working auth for demo. Creates/returns a user and a JWT."""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    user = db["user"].find_one({"email": body.email})
    if not user:
        model = User(
            name=body.name,
            email=body.email,
            role='restaurant' if body.role == 'restaurant' else 'customer',
            oauth_provider=None,
            oauth_id=None,
            avatar_url=body.avatar_url,
            is_active=True,
        )
        _ = create_document("user", model)
        user = db["user"].find_one({"email": body.email})

    # ensure a restaurant exists for restaurant role users (simple single-restaurant setup)
    role = 'restaurant' if body.role == 'restaurant' else 'customer'
    if role == 'restaurant':
        owner_id = str(user.get("_id"))
        rest = db["restaurant"].find_one({"owner_user_id": owner_id})
        if not rest:
            rest_model = Restaurant(name=f"{body.name}'s Kitchen", owner_user_id=owner_id)
            create_document("restaurant", rest_model)

    token = create_jwt({
        "sub": str(user.get("_id")),
        "email": user.get("email"),
        "name": user.get("name"),
        "role": role,
    })
    return {"token": token, "user": {"id": str(user.get("_id")), "email": user.get("email"), "name": user.get("name"), "role": role}}


# Optional Google OAuth endpoints (will 302 if not configured)
@app.get("/auth/providers")
def auth_providers():
    providers = []
    if OAUTH_AVAILABLE and os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
        providers.append("google")
    return {"providers": providers}


if OAUTH_AVAILABLE:
    try:
        starlette_config = StarletteConfig(environ=os.environ)
        oauth = OAuth(starlette_config)
        if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
            oauth.register(
                name='google',
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                client_kwargs={'scope': 'openid email profile'},
            )

        @app.get('/auth/login/google')
        async def login_via_google(request: Request):
            if 'google' not in oauth:
                raise HTTPException(status_code=400, detail='Google OAuth not configured')
            redirect_uri = request.url_for('auth_google_callback')
            return await oauth.google.authorize_redirect(request, redirect_uri)

        @app.get('/auth/callback/google')
        async def auth_google_callback(request: Request):
            if 'google' not in oauth:
                raise HTTPException(status_code=400, detail='Google OAuth not configured')
            token = await oauth.google.authorize_access_token(request)
            userinfo = token.get('userinfo')
            if not userinfo:
                raise HTTPException(status_code=400, detail='No userinfo from Google')
            # upsert user
            existing = db["user"].find_one({"email": userinfo['email']})
            if not existing:
                model = User(
                    name=userinfo.get('name') or userinfo.get('email'),
                    email=userinfo['email'],
                    role='customer',
                    oauth_provider='google',
                    oauth_id=userinfo.get('sub'),
                    avatar_url=userinfo.get('picture'),
                    is_active=True,
                )
                create_document('user', model)
                existing = db['user'].find_one({"email": userinfo['email']})
            token_str = create_jwt({
                "sub": str(existing.get("_id")),
                "email": existing.get("email"),
                "name": existing.get("name"),
                "role": existing.get("role", 'customer'),
            })
            # redirect back to frontend with token
            redirect_to = f"{FRONTEND_URL}/auth/callback?token={token_str}"
            return RedirectResponse(url=redirect_to)
    except Exception:
        pass


# ---------------------- Restaurants & Menu ----------------------
class UpsertRestaurant(BaseModel):
    name: str
    address: Optional[str] = None
    phone: Optional[str] = None


class UpsertMenuItem(BaseModel):
    name: str
    description: Optional[str] = None
    price_cents: int = Field(..., ge=0)
    is_available: bool = True
    image_url: Optional[str] = None


@app.get("/restaurants")
def list_restaurants():
    items = get_documents("restaurant")
    for it in items:
        it["id"] = str(it.pop("_id"))
    return items


@app.post("/restaurants")
def create_restaurant(body: UpsertRestaurant, user=Depends(get_current_user)):
    if user.get('role') != 'restaurant':
        raise HTTPException(status_code=403, detail="Only restaurant users can create restaurants")
    model = Restaurant(name=body.name, address=body.address, phone=body.phone, owner_user_id=user['sub'])
    rid = create_document('restaurant', model)
    return {"id": rid}


@app.get("/restaurants/{restaurant_id}/menu")
def get_menu(restaurant_id: str):
    items = get_documents("menuitem", {"restaurant_id": restaurant_id})
    for it in items:
        it["id"] = str(it.pop("_id"))
    return items


@app.post("/restaurants/{restaurant_id}/menu")
def add_menu_item(restaurant_id: str, body: UpsertMenuItem, user=Depends(get_current_user)):
    # In real app verify owner; demo keeps it open for a restaurant role user
    if user.get('role') != 'restaurant':
        raise HTTPException(status_code=403, detail="Only restaurant users can add menu items")
    model = MenuItem(restaurant_id=restaurant_id, name=body.name, description=body.description,
                     price_cents=body.price_cents, is_available=body.is_available, image_url=body.image_url)
    mid = create_document('menuitem', model)
    return {"id": mid}


# ---------------------- Orders & Real-time stream ----------------------
class PlaceOrderBody(BaseModel):
    restaurant_id: str
    items: List[OrderItem]
    note: Optional[str] = None


class UpdateOrderStatusBody(BaseModel):
    status: str = Field(..., pattern="^(pending|accepted|preparing|ready|completed|cancelled)$")


# In-memory broadcaster for SSE per restaurant (real-time only, not persisted)
import asyncio
from collections import defaultdict

restaurant_streams: Dict[str, List[asyncio.Queue]] = defaultdict(list)


def broadcast_event(restaurant_id: str, event: Dict[str, Any]):
    queues = restaurant_streams.get(restaurant_id, [])
    for q in list(queues):
        try:
            q.put_nowait(event)
        except Exception:
            pass


@app.post("/orders")
def place_order(body: PlaceOrderBody, user=Depends(get_current_user)):
    if user.get('role') != 'customer':
        raise HTTPException(status_code=403, detail="Only customers can place orders")
    # compute total from provided items
    total = 0
    for it in body.items:
        total += it.quantity * it.unit_price_cents
    order = Order(
        restaurant_id=body.restaurant_id,
        customer_id=user['sub'],
        items=body.items,
        note=body.note,
        total_cents=total,
        status='pending',
        placed_at=datetime.now(timezone.utc)
    )
    oid = create_document('order', order)
    event = OrderEvent(order_id=oid, type='new_order', message='New order placed').model_dump()
    broadcast_event(body.restaurant_id, event)
    return {"id": oid}


@app.get("/orders")
def list_orders(restaurant_id: Optional[str] = None, customer_id: Optional[str] = None):
    filt: Dict[str, Any] = {}
    if restaurant_id:
        filt['restaurant_id'] = restaurant_id
    if customer_id:
        filt['customer_id'] = customer_id
    items = get_documents('order', filt)
    for it in items:
        it['id'] = str(it.pop('_id'))
    return items


@app.patch("/orders/{order_id}/status")
def update_order_status(order_id: str, body: UpdateOrderStatusBody, user=Depends(get_current_user)):
    # In a real app verify restaurant ownership. For demo, allow if role is restaurant
    if user.get('role') != 'restaurant':
        raise HTTPException(status_code=403, detail="Only restaurant can update order status")
    # update by ObjectId
    try:
        _id = to_object_id(order_id)
    except HTTPException:
        raise HTTPException(status_code=400, detail="Invalid order id")
    res = db['order'].update_one({"_id": _id}, {"$set": {"status": body.status, "updated_at": datetime.now(timezone.utc)}})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Order not found")
    order = db['order'].find_one({"_id": _id})
    if order:
        event = OrderEvent(order_id=str(order.get('_id')), type='status_changed', status=body.status).model_dump()
        broadcast_event(order.get('restaurant_id'), event)
    return {"ok": True}


@app.get("/restaurants/{restaurant_id}/orders/stream")
async def stream_orders(restaurant_id: str, token: Optional[str] = None):
    """Server-Sent Events stream for restaurant dashboards.
    Token can be provided via query for auth (SSE doesn't send headers easily).
    """
    if token:
        _ = decode_jwt(token)  # validate
    queue: asyncio.Queue = asyncio.Queue()
    restaurant_streams[restaurant_id].append(queue)

    async def event_gen():
        try:
            # On connect, send a ping
            yield f"data: {json.dumps({'type':'ping','ts': datetime.now(timezone.utc).isoformat()})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            try:
                restaurant_streams[restaurant_id].remove(queue)
            except ValueError:
                pass

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------- Misc ----------------------
@app.get("/")
def read_root():
    return {"message": "Restaurant Ordering API"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
            response["database_name"] = os.getenv("DATABASE_NAME") or "❌ Not Set"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
