import socket
from typing import Optional

__internal_conn:Optional[socket.socket] = None

def __sanitize(v: str):
    for _ in range(25 - len(v)):
        v += "\0"

    return v

def __connect(addr: str, port: int) -> socket.socket:
    conn = socket.create_connection((addr, port))
    return conn

def send_movement(mov: str):
    bts = __sanitize(mov).encode()
    __internal_conn.sendall(bts)

def con():
    global __internal_conn
    __internal_conn = __connect("localhost", 7777)

def discon():
    global __internal_conn
    __internal_conn.close()