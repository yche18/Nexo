"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import json
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        keyword_store=None,
        allow_local_fallback=True,
        seed_defaults=True,
    ):
        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            if not allow_local_fallback:
                raise
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 使用服务端时不传 embedding_function，让服务端处理
        # 本地模式时也不传，使用 ChromaDB 默认的（会触发模型下载）
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "Nexo RAG 知识库"},
        )

        self._keyword_store = None
        self._import_lock = asyncio.Lock()

        # 如果知识库为空，导入默认文档
        self._seeded_default_docs = False
        if seed_defaults and self._collection.count() == 0:
            self._load_default_docs()
            self._seeded_default_docs = True

        self._keyword_store = keyword_store
        self._legacy_metadata = {
            item["chunk_id"]: dict(item, status="legacy")
            for item in self._legacy_updates(self._read_chunks())
        }
        if self._legacy_metadata:
            logger.warning("知识库含旧版元数据：暂用只读向量检索，请显式运行重建脚本完成迁移")

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def _prepare_document(self, document, preserve_legacy_ids=False):
        title = document.get("title", "")
        content = document.get("content", "")
        source = document.get("source", "")
        section = document.get("section", "")
        doc_id = document.get("doc_id") or hashlib.sha256(
            f"{title}\n{source}".encode()
        ).hexdigest()[:24]
        chunks = self._chunk_text(content, chunk_size=500)
        version_text = json.dumps([title, source, section, chunks], ensure_ascii=False)
        doc_version = hashlib.sha256(version_text.encode()).hexdigest()[:24]
        items = []
        for index, content in enumerate(chunks):
            if preserve_legacy_ids:
                chunk_id = hashlib.md5(f"{title}_{index}_{content[:50]}".encode()).hexdigest()
            else:
                chunk_id = hashlib.sha256(
                    f"{doc_id}:{doc_version}:{index}".encode()
                ).hexdigest()
            items.append({
                "doc_id": doc_id, "doc_version": doc_version, "chunk_id": chunk_id,
                "chunk_index": index, "total_chunks": len(chunks), "title": title,
                "source": source, "section": section, "content": content, "status": "pending",
            })
        return items

    def _upsert_chunks(self, chunks):
        if not chunks:
            return
        self._collection.upsert(
            ids=[item["chunk_id"] for item in chunks],
            documents=[item["content"] for item in chunks],
            metadatas=[self._metadata(item) for item in chunks],
        )

    @staticmethod
    def _metadata(item):
        keys = ("doc_id", "doc_version", "chunk_id", "chunk_index", "total_chunks",
                "title", "source", "section", "status")
        return {key: item[key] for key in keys}

    def _read_chunks(self, **kwargs):
        result = self._collection.get(include=["documents", "metadatas"], **kwargs)
        items = []
        for chunk_id, content, meta in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            item = dict(meta or {})
            item.update(chunk_id=chunk_id, content=content)
            items.append(item)
        return items

    def add_documents(self, documents: List[Dict[str, str]], preserve_legacy_ids=False) -> int:
        """同步写入用于无关键词库的场景；双库导入使用 add_documents_async。"""
        if getattr(self, "_legacy_metadata", {}):
            raise RuntimeError("旧知识库仍可检索；导入新版前请先显式迁移元数据")
        if getattr(self, "_keyword_store", None) is not None:
            raise RuntimeError("双库导入请使用 add_documents_async")
        total = 0
        for document in documents:
            chunks = self._prepare_document(document, preserve_legacy_ids)
            if not chunks:
                continue
            previous = self._read_chunks(where={"doc_id": chunks[0]["doc_id"]})
            same_version = {item["chunk_index"]: item for item in previous
                            if item["doc_version"] == chunks[0]["doc_version"]}
            for item in chunks:
                item["status"] = "ready"
                if item["chunk_index"] in same_version:
                    item["chunk_id"] = same_version[item["chunk_index"]]["chunk_id"]
            # 同一批次更新旧版状态与新版，检索只读取 ready。
            retired = []
            for item in previous:
                if item["doc_version"] != chunks[0]["doc_version"]:
                    retired.append(dict(item, status="retired"))
            self._upsert_chunks(retired + chunks)
            total += len(chunks)
        return total

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """先写 pending，ES 成功后发布；失败可用相同输入重试，不视为跨库事务。"""
        if getattr(self, "_legacy_metadata", {}):
            raise RuntimeError("旧知识库仍可检索；导入新版前请先显式迁移元数据")
        if self._keyword_store is None:
            return await asyncio.to_thread(self.add_documents, documents)
        async with self._import_lock:
            total = 0
            for document in documents:
                chunks = self._prepare_document(document)
                if not chunks:
                    continue
                doc_id = chunks[0]["doc_id"]
                version = chunks[0]["doc_version"]
                previous = await asyncio.to_thread(self._read_chunks, where={"doc_id": doc_id})
                same_version = {item["chunk_index"]: item for item in previous
                                if item["doc_version"] == version and item["status"] == "ready"}
                # 元数据迁移后，重复导入仍沿用旧 chunk ID。
                for item in chunks:
                    if item["chunk_index"] in same_version:
                        item["chunk_id"] = same_version[item["chunk_index"]]["chunk_id"]
                already_ready = len(same_version) == len(chunks)
                ready = [dict(item, status="ready") for item in chunks]
                retired = [dict(item, status="retired") for item in previous
                           if item["doc_version"] != version]
                if not already_ready:
                    await asyncio.to_thread(self._upsert_chunks, chunks)
                    await self._keyword_store.index_chunks(chunks)
                # ES 提前 ready 的短暂窗口由 keyword_search_handler 的 Chroma 状态校验隔离。
                await self._keyword_store.index_chunks(ready)
                if not already_ready:
                    updates = retired + ready
                    try:
                        await asyncio.to_thread(
                            self._collection.update,
                            ids=[item["chunk_id"] for item in updates],
                            metadatas=[self._metadata(item) for item in updates],
                        )
                    except Exception:
                        # 恢复旧版可见状态，新版保留 pending，便于重试。
                        rollback = previous + [item for item in chunks
                                               if item["chunk_id"] not in {old["chunk_id"] for old in previous}]
                        await asyncio.to_thread(
                            self._collection.update,
                            ids=[item["chunk_id"] for item in rollback],
                            metadatas=[self._metadata(item) for item in rollback],
                        )
                        raise
                old_versions = {item["doc_version"] for item in retired}
                for old_version in old_versions:
                    await self._keyword_store.delete_document_chunks(doc_id, old_version)
                total += len(chunks)
            return total

    def _legacy_updates(self, chunks):
        """构造旧切片元数据，不修改数据库。"""
        groups = {}
        for item in chunks:
            if "doc_version" not in item:
                key = (item.get("title", ""), item.get("source", ""))
                groups.setdefault(key, []).append(item)
        changed = []
        for (title, source), items in groups.items():
            items.sort(key=lambda item: item.get("chunk_index", 0))
            indexes = [item.get("chunk_index", 0) for item in items]
            if len(set(indexes)) != len(indexes):
                raise ValueError(f"旧文档 {title!r} 有重复 chunk_index，请确认版本后再迁移")
            doc_id = hashlib.sha256(f"{title}\n{source}".encode()).hexdigest()[:24]
            content = [item["content"] for item in items]
            version_text = json.dumps([title, source, "", content], ensure_ascii=False)
            version = hashlib.sha256(version_text.encode()).hexdigest()[:24]
            for item in items:
                item.update(doc_id=doc_id, doc_version=version, title=title,
                            source=source, section=item.get("section", ""), status="ready",
                            chunk_index=item.get("chunk_index", 0), total_chunks=len(items))
                changed.append(item)
        return changed

    def migrate_legacy_metadata(self) -> int:
        """显式补齐旧知识库元数据，保留原 ID、文本及切片边界，不访问记忆集合。"""
        changed = self._legacy_updates(self._read_chunks())
        if changed:
            self._collection.update(
                ids=[item["chunk_id"] for item in changed],
                metadatas=[self._metadata(item) for item in changed],
            )
        self._legacy_metadata = {}
        return len(changed)

    async def initialize_keyword_index(self):
        """新索引/新种子或空索引恢复时补建；旧元数据仍要求显式迁移。"""
        if self._keyword_store is None:
            return
        created = await self._keyword_store.ensure_index()
        if self._legacy_metadata:
            logger.warning("Elasticsearch 尚未同步旧知识切片，请运行显式迁移重建脚本")
            return
        empty = await self._keyword_store.count_ready() == 0
        if created or getattr(self, "_seeded_default_docs", False) or (empty and self.doc_count):
            await self.rebuild_keyword_index()

    async def rebuild_keyword_index(self) -> int:
        """显式重建 ready 知识切片的关键词索引，可重复执行。"""
        if self._keyword_store is None:
            raise RuntimeError("未配置 Elasticsearch")
        async with self._import_lock:
            await self._keyword_store.ensure_index()
            chunks = await asyncio.to_thread(self._read_chunks)
            if any("doc_version" not in item for item in chunks):
                raise ValueError("旧知识库缺少元数据，请先执行 migrate_legacy_metadata")
            ready = [item for item in chunks if item["status"] == "ready"]
            await self._keyword_store.index_chunks(ready)
            for item in chunks:
                if item["status"] != "ready":
                    await self._keyword_store.delete_document_chunks(item["doc_id"], item["doc_version"])
            return len(ready)

    @staticmethod
    def _candidate(item, score=0.0, channel="vector"):
        result = dict(item)
        result.update(score=score, chunk=item["chunk_index"], channel=channel)
        return result

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """只检索已发布知识切片；保留原 Chroma 距离到 score 的计算方式。"""
        if top_k <= 0 or self._collection.count() == 0:
            return []
        legacy = getattr(self, "_legacy_metadata", {})
        kwargs = {"where": {"status": "ready"}}
        count = min(top_k, self._collection.count())
        if legacy:
            # 仅迁移前兼容旧记录；pending/retired 仍在下面过滤。
            kwargs = {}
            count = self._collection.count()
        results = self._collection.query(query_texts=[query], n_results=count, **kwargs)
        items = []
        for chunk_id, content, meta, distance in zip(
            results["ids"][0], results["documents"][0],
            results["metadatas"][0], results["distances"][0],
        ):
            item = dict(meta)
            if "doc_version" not in item and chunk_id in legacy:
                item = dict(legacy[chunk_id])
            if item.get("status") not in ("ready", "legacy"):
                continue
            item.update(chunk_id=chunk_id, content=content)
            items.append(self._candidate(item, round(1.0 - distance, 4)))
        return items[:top_k]

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        ready = self._collection.get(where={"status": "ready"}, include=[])["ids"]
        return len(set(ready) | set(getattr(self, "_legacy_metadata", {})))

    async def doc_count_async(self) -> int:
        return await asyncio.to_thread(lambda: self.doc_count)

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        return await self.search_async(params.get("query", ""), params.get("top_k", 5))

    async def keyword_search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        if self._keyword_store is None:
            raise RuntimeError("未配置 Elasticsearch")
        top_k = params.get("top_k", 5)
        if top_k <= 0:
            return []
        items = await self._keyword_store.search(params.get("query", ""), top_k * 3)
        if not items:
            return []
        published = await asyncio.to_thread(
            self._read_chunks, ids=[item["chunk_id"] for item in items], where={"status": "ready"}
        )
        valid = {item["chunk_id"]: item for item in published}
        result = []
        for item in items:
            current = valid.get(item["chunk_id"])
            if current and current["doc_version"] == item["doc_version"]:
                # 正文与元数据以 Chroma 已发布版本为准，ES 只提供排序。
                result.append(self._candidate(current, item["score"], "keyword"))
        return result[:top_k]

    async def get_neighbors_async(self, item, window=1):
        """返回同文档、版本、章节的 ready 邻接切片（包含命中切片），按序排列。"""
        where = {"$and": [
            {"doc_id": item["doc_id"]}, {"doc_version": item["doc_version"]},
            {"section": item.get("section", "")}, {"status": "ready"},
        ]}
        chunks = await asyncio.to_thread(self._read_chunks, where=where)
        neighbors = []
        for chunk in chunks:
            if abs(chunk["chunk_index"] - item["chunk_index"]) <= max(0, window):
                neighbors.append(self._candidate(chunk))
        return sorted(neighbors, key=lambda chunk: chunk["chunk_index"])

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """保留原按句号/换行切片；单句超长时按长度兜底，避免超过上限。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []
        chunks = []
        current = ""
        for sentence in text.replace("\n", "。").split("。"):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                while len(sentence) > chunk_size:
                    chunks.append(sentence[:chunk_size])
                    sentence = sentence[chunk_size:]
            if len(current) + len(sentence) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current}。{sentence}" if current else sentence
        if current:
            chunks.append(current)
        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（IT 支持演示资料）。"""
        default_docs = [
            {
                "title": "VPN 连接排查（演示资料）",
                "content": (
                    "以下为模拟企业 IT 资料。"
                    "VPN 连接失败时，先确认普通网页能否访问，区分基础网络故障和 VPN 连接问题。"
                    "记录操作系统、VPN 客户端版本、错误提示、发生时间和是否持续复现。"
                    "401 通常表示认证失败，可检查登录状态、是否使用正确入口，以及认证信息是否过期。"
                    "连接超时时，按网络、DNS、代理、防火墙和服务状态顺序排查，不直接断定根因。"
                    "如果切换网络或重新登录后仍失败，保留错误信息和已尝试的步骤，联系 IT 人员继续核验。"
                    "不要发送密码、验证码或完整密钥，不绕过认证和网络访问限制。"
                ),
            },
            {
                "title": "权限申请流程（演示 SOP）",
                "content": (
                    "以下为模拟企业流程，不代表真实公司政策。"
                    "申请项目系统或 VPN 访问权限时，准备目标系统名称和申请用途。"
                    "如果已明确所需权限范围，可一并说明只读、读写或项目访问需求，不填入猜测。"
                    "由负责人确认业务需要，再由 IT 按企业审批流程处理；"
                    "具体入口、所需材料和时效以实际 IT 规范为准。"
                    "Nexo 只检查目标系统和用途是否提供，信息齐全不代表身份已核验或审批已通过。"
                    "Nexo 不代为提交申请、不查询审批进度，也不执行授权或撤权。"
                    "若收到批准通知后仍无法访问，保留通知摘要和错误提示，联系 IT 核实实际开通状态。"
                ),
            },
            {
                "title": "企业账号安全（演示资料）",
                "content": (
                    "企业账号安全指引。"
                    "忘记密码时，使用企业指定的自助入口或联系 IT 核验，不通过非官方渠道重置。"
                    "发现异常登录时记录时间、涉及系统和问题现象，及时联系 IT 人员核查。"
                    "账号被锁定、登录方式变更或账号归属不明时，由 IT 按企业流程核验。"
                    "是否可以开启两步验证以及如何配置，以企业账号安全规范为准。"
                    "不要共享账号，不向助手或他人发送密码、验证码或完整密钥。"
                    "Nexo 无法查询账号是否锁定、验证员工身份或执行密码重置，仅提供说明和交接材料。"
                ),
            },
            {
                "title": "办公软件安装（演示 SOP）",
                "content": (
                    "以下为模拟企业软件安装流程。"
                    "先确认软件名称、操作系统、所需版本和希望完成的工作。"
                    "查看企业软件目录，或联系 IT 获取已批准的安装包与适用操作指引。"
                    "需要管理员权限或许可证时，按企业流程申请，不绕过安装限制或使用来源不明的安装包。"
                    "安装前保存当前工作，涉及覆盖配置或已有数据时先按企业规范确认备份。"
                    "安装失败时记录错误提示、发生时间及已尝试的步骤，交给技术支持排查。"
                    "Nexo 不会直接安装软件、查询许可证或修改设备权限；"
                    "具体可用软件和授权范围需要 IT 确认。"
                ),
            },
            {
                "title": "办公软件与网页故障排查（演示资料）",
                "content": (
                    "办公软件与网页常见故障排查。"
                    "软件闪退时记录软件和系统版本、错误提示、发生时间及最近变更，先尝试保存工作后重启应用。"
                    "登录失败 401 通常表示认证失败，应核对登录状态和认证信息是否过期，不发送密码或完整 Token。"
                    "网页加载慢时检查网络连接，并记录是否只影响某个页面、某个用户或多个用户。"
                    "网页 500 表示服务端处理异常，记录请求时间、接口路径和 request_id，联系 IT 继续核验。"
                    "先收集信息再判断原因，不直接把问题归咎于用户或某个内部组件。"
                    "如果影响多人或核心业务，应整理影响范围和排查记录转人工；"
                    "不直接删除数据或重置生产配置。"
                ),
            },
            {
                "title": "IT 人工交接（演示 SOP）",
                "content": (
                    "IT 人工交接说明。"
                    "用户要求人工、涉及实际权限操作，或按现有步骤仍不能处理时，整理信息供 IT 人员核验。"
                    "问题说明应包含涉及系统、问题现象、发生时间、影响范围和用户期望。"
                    "技术问题附上环境、错误提示、日志摘要和已经尝试的步骤，避免重复排查。"
                    "权限申请附上目标系统、申请用途，以及用户明确提供的权限范围。"
                    "交接时不要包含密码、验证码、完整密钥等敏感内容，缺少信息时说明尚待补充。"
                    "Nexo 的交接说明只供员工联系 IT 时使用，尚未创建工单、派单或通知人员，不承诺处理进度与完成时间。"
                ),
            },
        ]
        self.add_documents(default_docs, preserve_legacy_ids=True)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
