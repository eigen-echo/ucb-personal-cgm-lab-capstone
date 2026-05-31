import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.environ.get("DB_PATH", "/app/data/cgm_app.db")

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Import all models so Base sees them before create_all
    from app.models import cgm, meal, activity, medication, fasting_window, food_item, model_run, spike_event, app_config, glucose_prediction  # noqa: F401
    Base.metadata.create_all(bind=engine)
