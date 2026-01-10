from sqlalchemy import create_engine, Column, Integer, String, Text, JSON, DateTime, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from pgvector.sqlalchemy import Vector
from datetime import datetime
from config import DATABASE_URL # Tu fix del import

Base = declarative_base()

class Review(Base):
    __tablename__ = 'reviews'

    # Metadatos básicos
    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String)         # "Reddit", "Trustpilot"
    url = Column(String, unique=True) # Evitar duplicados
    author = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Contenido
    raw_content = Column(Text, nullable=False)
    cleaned_content = Column(Text)  # Para cumplir requisitos NLP académicos

    # --- INTELIGENCIA DE NEGOCIO (Extraction by Granite 3.3) ---

    # Sentimiento Avanzado
    sentiment_label = Column(String) # "Positivo", "Negativo", "Mixto"
    sentiment_score = Column(Float)  # -1.0 a 1.0

    # Análisis Cualitativo
    emotions = Column(JSON)          # Ej: ["Ira", "Frustración"]
    topics = Column(JSON)            # Ej: ["Precio", "Atención al Cliente"]
    pain_points = Column(JSON)       # Ej: ["Tiempo de espera", "Comisiones ocultas"]
    competitors = Column(JSON)       # Ej: ["Santander", "BBVA"]

    # Flags de Riesgo
    is_churn_risk = Column(Boolean)  # ¿Dice que se va a dar de baja?
    is_bot_suspect = Column(Boolean) # ¿Parece spam?

    # --- VECTOR STORE ---
    # Granite Embedding 278m usa 768 dimensiones.
    embedding = Column(Vector(768))

def init_db():
    print(f"🔌 Conectando a DB en: {DATABASE_URL} ...")
    engine = create_engine(DATABASE_URL)

    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        Base.metadata.create_all(engine)
        print("Esquema desplegado (Vector 768 dims).")
        return sessionmaker(bind=engine)()

    except Exception as e:
        print(f"Error DB: {e}")
        return None

if __name__ == "__main__":
    init_db()