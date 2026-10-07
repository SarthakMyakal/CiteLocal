"""Engine and session factory."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db.models import Base


def make_session_factory(database_url: str) -> sessionmaker:
    """Create the engine, create tables if missing, and return a session factory."""
    # check_same_thread=False: FastAPI runs sync endpoints in a thread pool.
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)
