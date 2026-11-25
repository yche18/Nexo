"""显式迁移知识切片元数据并重建 ES：在 backend 目录运行 python -m mcp.rebuild_knowledge_index。"""
import argparse
import asyncio
import os

from dotenv import load_dotenv

from mcp.elasticsearch_store import ElasticsearchStore
from mcp.knowledge_base import KnowledgeBase


async def main():
    parser = argparse.ArgumentParser(description="仅迁移 knowledge_base，不访问任何记忆集合")
    parser.add_argument("--migrate-legacy", action="store_true", help="保留旧切片 ID 和正文，补齐版本元数据")
    parser.add_argument("--allow-local-fallback", action="store_true", help="明确允许连接失败时迁移 CHROMA_PERSIST_DIRECTORY 本地知识库")
    args = parser.parse_args()
    load_dotenv()
    store = ElasticsearchStore(
        username=os.getenv("ELASTICSEARCH_USERNAME"), password=os.getenv("ELASTICSEARCH_PASSWORD"),
        api_key=os.getenv("ELASTICSEARCH_API_KEY"),
    )
    try:
        kb = KnowledgeBase(
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma"), keyword_store=store,
            allow_local_fallback=args.allow_local_fallback, seed_defaults=False,
        )
        if args.migrate_legacy:
            migrated = await asyncio.to_thread(kb.migrate_legacy_metadata)
            print(f"元数据迁移完成：{migrated} 个切片，原 ID 和正文保留")
        count = await kb.rebuild_keyword_index()
        print(f"关键词索引重建完成：{count} 个 ready 切片")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
