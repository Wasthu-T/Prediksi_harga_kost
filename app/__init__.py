from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Inisiasi Flask
app = Flask(__name__)
limiter = Limiter(
    get_remote_address,  # Identifikasi pengguna berdasarkan alamat IP
    app=app,
    default_limits=["5 per minute"]  # Membatasi 5 permintaan per menit
    
)   
from app import routes