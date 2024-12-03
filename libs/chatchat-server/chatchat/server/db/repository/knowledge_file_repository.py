from chatchat.server.db.session import with_async_session, async_session_scope
from sqlalchemy import select, delete
from chatchat.server.db.model.knowledge_base_model import KnowledgeBaseModel
from chatchat.server.db.model.knowledge_file_model import KnowledgeFileModel, FileDocModel
from chatchat.server.knowledge_base.utils import KnowledgeFile
from typing import List, Dict


@with_async_session
async def list_file_num_docs_id_by_kb_name_and_file_name(
        session,
        kb_name: str,
        file_name: str,
) -> List[int]:
    doc_ids = session.query(FileDocModel.doc_id).filter_by(
        kb_name=kb_name,
        file_name=file_name,
    ).all()
    return [int(_id[0]) for _id in doc_ids]


@with_async_session
async def list_docs_from_db(
        session,
        kb_name: str,
        file_name: str,
        metadata: Dict = {},
) -> List[Dict]:
    # docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    if file_name:
        # docs = docs.filter(FileDocModel.file_name.ilike(file_name))
        docs = (await session.execute(
            select(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name)).filter(
                FileDocModel.file_name.ilike(file_name))
        )).scalars()
    for k, v in metadata.items():
        # docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))
        docs = (await session.execute(
            select(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name)).filter(
                FileDocModel.meta_data[k].as_string() == str(v))
        )).scalars()
    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_async_session
async def delete_docs_from_db(
        session,
        kb_name: str,
        file_name: str = None
) -> List[Dict]:
    docs = await list_docs_from_db(kb_name=kb_name,
                                   file_name=file_name)
    if file_name:
        await session.execute(
            delete(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name),
                                        FileDocModel.file_name.ilike(file_name))
        )
    else:
        await session.execute(
            delete(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
        )
    # query = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))
    # if file_name:
    #     query = query.filter(FileDocModel.file_name.ilike(file_name))
    await session.commit()
    return docs


@with_async_session
async def add_docs_to_db(session,
                         kb_name: str,
                         file_name: str,
                         doc_infos: List[Dict]):
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None")
        return False
    try:
        for doc_info in doc_infos:
            obj = FileDocModel(
                kb_name=kb_name,
                file_name=file_name,
                doc_id=doc_info['id'],
                meta_data=str(doc_info['metadata']),
            )
            session.add(obj)
        await session.commit()
        print("文档信息成功添加到数据库")
        return True
    except Exception as e:
        print(f"添加文档发生错误: {e}")
        await session.rollback()
        return False


@with_async_session
async def count_files_from_db(session,
                              kb_name: str) -> int:
    return session.query(KnowledgeFileModel).filter(
        KnowledgeFileModel.kb_name.ilike(kb_name)
    ).count()


@with_async_session
async def list_files_from_db(session,
                             kb_name):
    files = (await session.execute(
        select(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name))
    )).scalars().all()
    # files = session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).all()
    docs = [f.file_name for f in files]
    return docs


@with_async_session
async def delete_file_from_db(session,
                              kb_file: KnowledgeFile):
    existing_file = (await session.execute(
        select(KnowledgeFileModel)
        .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
    )).scalars().first()

    if existing_file:
        # 删除文件
        await session.delete(existing_file)
        await delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        await session.commit()

        # 异步查询关联的知识库
        kb = (await session.execute(
            select(KnowledgeBaseModel)
            .filter(KnowledgeBaseModel.kb_name.ilike(kb_file.kb_name))
        )).scalars().first()

        if kb:
            kb.file_count -= 1
            await session.commit()

    return True


@with_async_session
async def add_file_to_db(session,
                         kb_file: KnowledgeFile,
                         docs_count: int = 0,
                         custom_docs: bool = False,
                         doc_infos: List[Dict] = [],
                         ):
    print("开始查询 KnowledgeBase")
    stmt = select(KnowledgeBaseModel).where(KnowledgeBaseModel.kb_name == kb_file.kb_name)
    kb_result = await session.execute(stmt)
    kb = kb_result.scalars().first()
    print(f"查询Kb 完成: {kb}")

    if kb:
        print("kb存在，开始查询kb")
        stmt = select(KnowledgeFileModel).where(
            KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
            KnowledgeFileModel.file_name.ilike(kb_file.filename)
        )
        file_result = await session.execute(stmt)
        existing_file = file_result.scalars().first()
        print(f"查询kb 完成 {existing_file}")
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()
        print(f"获取文件事件和大小: mtime={mtime}, size={size}")
        if existing_file:
            print("文件存在，更新文件信息...")
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
            print("文件信息更新完成")
        else:
            print("文件不存在，创建新文件...")
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            kb.file_count += 1
            session.add(new_file)
            print("新文件添加完成")
        print("开始添加文档信息")
        await add_docs_to_db(
            kb_name=kb_file.kb_name,
            file_name=kb_file.filename,
            doc_infos=doc_infos
        )
        print("文件信息添加完成")

        try:
            print("开始提交事务")
            await session.commit()
            print("事务提交成功")
        except Exception as e:
            print(f"Error commiting changes {e}")
            await session.rollback()
            print("事务回滚")
            raise
    else:
        print("Kb 不存在，无法添加文件")
    return True


@with_async_session
async def file_exists_in_db(session, kb_file: KnowledgeFile):
    existing_file = (await session.execute(
        select(KnowledgeFileModel).filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                                          KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
    )).scalars().first()
    # existing_file = (session.query(KnowledgeFile)
    #                  .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
    #                          KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
    #                  .first()
    #                  )
    return True if existing_file else False


@with_async_session
async def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (await session.execute(
        select(KnowledgeFileModel).filter(KnowledgeFileModel.file_name.ilike(filename),
                                          KnowledgeFileModel.kb_name.ilike(kb_name))
    )).scalars().first()
    # file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
    #                             .filter(KnowledgeFileModel.file_name.ilike(filename),
    #                                     KnowledgeFileModel.kb_name.ilike(kb_name))
    #                             .first()
    #                             )
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}


@with_async_session
async def delete_files_from_db(session, knowledge_base_name: str):
    await session.execute(
        delete(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(knowledge_base_name))
    )
    await session.execute(
        delete(FileDocModel).filter(FileDocModel.kb_name.ilike(knowledge_base_name))
    )
    kb = (await session.execute(
        select(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name))
    )).scalars().first()

    if kb:
        kb.file_count = 0

    await session.commit()
    return True
