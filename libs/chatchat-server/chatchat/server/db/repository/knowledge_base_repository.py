from chatchat.server.db.model.knowledge_base_model import KnowledgeBaseModel, KnowledgeBaseSchema
from sqlalchemy import select, delete
from chatchat.server.db.session import with_async_session, async_session_scope
import asyncio


@with_async_session
async def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model, user_id):
    kb = await session.execute(
        select(KnowledgeBaseModel)
        .where(KnowledgeBaseModel.kb_name.ilike(kb_name))
    )
    kb = kb.scalars().first()

    if not kb:
        kb = KnowledgeBaseModel(
            kb_name=kb_name,
            kb_info=kb_info,
            vs_type=vs_type,
            embed_model=embed_model,
            user_id=user_id
        )
        session.add(kb)
    else:
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
        kb.user_id = user_id

    await session.commit()
    return True


@with_async_session
async def list_kbs_from_db(session, min_file_count: int = -1):
    result = await session.execute(
        select(KnowledgeBaseModel.kb_name)
        .where(KnowledgeBaseModel.file_count > min_file_count)
    )

    kbs = [kb for kb in result.scalars().all()]
    return kbs

@with_async_session
async def list_kbs_from_db2(session, min_file_count: int = -1):
    result = await session.execute(
        select(KnowledgeBaseModel)
        .where(KnowledgeBaseModel.file_count > min_file_count)
    )

    kbs = [KnowledgeBaseSchema.model_validate(kb) for kb in result.scalars().all()]
    return kbs



@with_async_session
async def kb_exists(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
    status = True if kb else False
    return status


@with_async_session
async def get_kb_detail(session, kb_name: str) -> dict:
    stmt = select(
        KnowledgeBaseModel
    ).where(KnowledgeBaseModel.kb_name.ilike(kb_name))
    result = await session.execute(stmt)
    kb = result.scalars().first()

    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}


@with_async_session
async def delete_kb_from_db(session, kb_name):
    await session.execute(
        delete(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
    )
    return True


@with_async_session
async def load_kb_from_db(session, kb_name):
    stmt = select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_name.ilike(kb_name))
    result = await session.execute(stmt)
    kb = result.scalar_one_or_none()

    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


if __name__ == '__main__':
    r = asyncio.run(list_kbs_from_db())
    print(r)


if __name__ == '__main__':
    r = asyncio.run(list_kbs_from_db())
    print(r)
