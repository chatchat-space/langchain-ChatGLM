import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from NorthinfoChat.server.db.base import Base

username = ''
hostname = ''
database_name = ''
password = ""

SQLALCHEMY_DATABASE_URI = f"mysql+asyncmy://{username}:{password}@{hostname}/{database_name}?charset=utf8mb4"

async_engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=True
)


async def drop_tables(engine):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


if __name__ == '__main__':
    asyncio.run(drop_tables(async_engine))

