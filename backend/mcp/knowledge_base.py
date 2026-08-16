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

        # 如果知识库为空，导入默认文档
        if self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # ID 由文档内容稳定生成；upsert 使重复导入幂等，同时允许同 ID 内容更新。
            # ChromaDB 会自动生成或更新 Embedding。
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库写入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        ChromaDB 内部自动将 query 转为向量，与存储的文档向量做余弦相似度匹配。
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

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
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
