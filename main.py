import time
import random
import uuid
import asyncio
from typing import Dict, Any
from pathlib import Path
from urllib.parse import urlparse

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
import astrbot.api.message_components as Comp


@register(
    "gpt_sovits_tts_local",
    "Amnemon",
    "支持本地 GPT-SoVITS 的文字转语音插件",
    "1.7.2"
)
class GPTSoVITSTTSLocal(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)

        model_prefix = config.get('model_prefix', '【GSVI】')
        if model_prefix == '无前缀':
            model_prefix = ''
        raw_gpt = config.get('gpt_model_name', '')
        raw_sovits = config.get('sovits_model_name', '')
        self.gpt_model_name = f"{model_prefix}{raw_gpt}" if raw_gpt else ''
        self.sovits_model_name = f"{model_prefix}{raw_sovits}" if raw_sovits else ''
        self.character_name = config.get('character_name', '').strip()
        self.version = config.get('version', 'v4')
        self.ref_audio_path = config.get('ref_audio_path', '')
        self.prompt_text = config.get('prompt_text', '')
        self.prompt_text_lang = config.get('prompt_text_lang', '中文')
        self.text_lang = config.get('text_lang', '中英混合')

        self.prob = float(config.get('prob', 100.0)) / 100.0
        self.text_limit = int(config.get('text_limit', 0))
        self.cooldown = int(config.get('cooldown', 0))
        self.send_text_with_audio = bool(config.get('send_text_with_audio', False))

        self.top_k = int(config.get('top_k', 10))
        self.top_p = float(config.get('top_p', 1.0))
        self.temperature = float(config.get('temperature', 1.0))
        self.speed_facter = float(config.get('speed_facter', 1.0))
        self.split_sentence = bool(config.get('split_sentence', True))

        raw_host = config.get('api_host', 'localhost').strip()
        # 兼容用户误填协议前缀的情况，例如 "http://127.0.0.1"、"https://127.0.0.1"、
        # 或漏打冒号的 "http//127.0.0.1"，统一清洗成纯 host，避免拼出
        # "http://http://xxx" 这种错误的 base_url 导致 502。
        for prefix in ("https://", "http://", "https//", "http//"):
            if raw_host.lower().startswith(prefix):
                raw_host = raw_host[len(prefix):]
                break
        # 去除误填的末尾斜杠
        raw_host = raw_host.rstrip('/')
        self.api_host = raw_host

        self.api_port = int(config.get('api_port', 9880))
        self.outputs_path = config.get('outputs_path', '').strip()
        self.tts_timeout = int(config.get('tts_timeout', 5)) * 60  # 转换为秒

        self.base_url = f"http://{self.api_host}:{self.api_port}"
        self.temp_dir = Path(__file__).parent / "temp_audio"
        self.temp_dir.mkdir(exist_ok=True)

        self._session_state: Dict[str, Any] = {}

        logger.info(f"[GPTSoVITSTTSLocal] 插件已加载，TTS 服务地址: {self.base_url}")
        char_tag = f" | 角色: {self.character_name}" if self.character_name else ""
        logger.info(f"[GPTSoVITSTTSLocal] GPT模型: {self.gpt_model_name}, SoVITS模型: {self.sovits_model_name}{char_tag}")

    async def _post(self, endpoint: str, data: dict, return_json=True):
        """
        统一的 POST 请求封装。

        return_json=True  -> 尝试解析并返回 JSON dict；解析失败则返回原始 bytes
        return_json=False -> 不尝试解析 JSON，直接返回原始 bytes（用于已知返回二进制流的接口）

        无论哪种模式，都会先检查 HTTP 状态码，非 200 直接抛异常，
        避免把错误响应（空 body / 错误 JSON）误当成正常数据处理。
        """
        import httpx
        try:
            logger.debug(f"[GPTSoVITSTTSLocal] 正在请求 {self.base_url}{endpoint}，参数: {data}")
            async with httpx.AsyncClient(timeout=self.tts_timeout) as client:
                resp = await client.post(f"{self.base_url}{endpoint}", json=data)

                if resp.status_code != 200:
                    logger.error(
                        f"[GPTSoVITSTTSLocal] TTS接口返回异常状态码 {resp.status_code}: "
                        f"{resp.text[:500]}"
                    )
                    resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")
                logger.debug(
                    f"[GPTSoVITSTTSLocal] 响应 content-type: {content_type}, "
                    f"content-length: {len(resp.content)} bytes"
                )

                if not return_json:
                    return resp.content

                # 尝试解析 JSON；如果 content-type 明显是音频或解析失败，返回原始 bytes
                if "audio" in content_type or "octet-stream" in content_type:
                    return resp.content

                try:
                    json_data = resp.json()
                    logger.debug(f"[GPTSoVITSTTSLocal] 收到 JSON 响应: {json_data}")
                    return json_data
                except Exception:
                    # 解析 JSON 失败，说明实际返回的是二进制流
                    return resp.content

        except Exception as e:
            logger.error(f"[GPTSoVITSTTSLocal] 请求失败 {endpoint}: {e}")
            raise

    async def _download_audio(self, url: str) -> bytes:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.error(f"[GPTSoVITSTTSLocal] 下载音频失败 {url}: {e}")
            return None

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """
        LLM 回复后触发，进行 TTS 转换
        """
        result = event.get_result()
        if not result or not result.chain:
            return

        # 提取纯文本部分并记录索引
        full_text = ""
        plain_indices = []
        for i, comp in enumerate(result.chain):
            if isinstance(comp, Comp.Plain):
                full_text += comp.text
                plain_indices.append(i)

        full_text = full_text.strip()
        if not full_text:
            return

        # 概率触发
        if random.random() > self.prob:
            return

        # 检查冷却
        session_id = event.session_id
        now = time.time()
        last_time = self._session_state.get(session_id, {}).get("last_trigger", 0)
        if self.cooldown > 0 and now - last_time < self.cooldown:
            return

        # 记录触发时间
        if session_id not in self._session_state:
            self._session_state[session_id] = {}
        self._session_state[session_id]["last_trigger"] = now

        # 文本截断
        tts_text = full_text
        if self.text_limit > 0 and len(tts_text) > self.text_limit:
            logger.warning(f"[GPTSoVITSTTSLocal] 文本过长，截断前 {self.text_limit} 个字符")
            tts_text = tts_text[:self.text_limit]

        logger.info(f"[GPTSoVITSTTSLocal] 开始转换语音: {tts_text[:20]}...")

        # 语言标签映射：兼容配置里填的中文标签，转换成 GPT-SoVITS api_v2.py
        # 官方 /tts 接口要求的语言代码（zh/en/ja/ko/yue/auto 等）。
        # 如果配置里填的已经是官方代码（比如用户直接填了 "zh"），也原样透传。
        _lang_map = {
            "中文": "zh", "英文": "en", "日文": "ja", "韩文": "ko",
            "粤语": "yue", "中英混合": "zh", "自动": "auto", "多语种混合": "auto",
        }
        text_lang_code = _lang_map.get(self.text_lang, self.text_lang)
        prompt_lang_code = _lang_map.get(self.prompt_text_lang, self.prompt_text_lang)

        # 切分方式：api_v2.py 只接受 cut0~cut5 这类英文枚举值，不接受中文描述
        # cut5 = 按标点符号切；cut0 = 不切
        text_split_method = "cut5" if self.split_sentence else "cut0"

        # 构造请求参数（对应 GPT_SoVITS/api_v2.py 中的 TTS_Request 字段）
        # 注意：api_v2.py 的 /tts 接口不支持在请求体里传 gpt_model_name /
        # sovits_model_name 来切换模型 —— 模型是服务启动时（或通过
        # /set_gpt_weights、/set_sovits_weights 接口）加载好的，这里不再传递。
        payload = {
            "text": tts_text,
            "text_lang": text_lang_code,
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.prompt_text,
            "prompt_lang": prompt_lang_code,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": text_split_method,
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": self.speed_facter,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": 16,
        }

        try:
            audio_url = None

            # GPT-SoVITS api_v2.py 的 /tts 接口固定直接返回二进制 wav 流
            # （StreamingResponse），不会返回 JSON + audio_url。
            # 这里仍然保留对旧版 audio_url 协议的兼容分支，以防你切换回旧版 api.py。
            resp = await self._post("/tts", payload, return_json=False)

            if isinstance(resp, bytes):
                # 服务端直接返回了二进制音频流
                if not resp:
                    logger.error("[GPTSoVITSTTSLocal] TTS 接口返回了空的二进制数据（0 字节），请检查 GPT-SoVITS 服务端日志")
                    return
                audio_data = resp
            else:
                # 兼容旧版：返回 JSON + audio_url
                if not isinstance(resp, dict):
                    logger.error(f"[GPTSoVITSTTSLocal] TTS 接口返回了非预期格式: {type(resp)}")
                    return

                audio_url = resp.get("audio_url")
                if not audio_url:
                    logger.error(f"[GPTSoVITSTTSLocal] TTS 接口未返回 audio_url: {resp}")
                    return

                # 如果配置的是 0.0.0.0，返回的 url 可能是 0.0.0.0:port，
                # 这在部分系统上可能无法直接连接，替换为配置的 host
                audio_url = audio_url.replace("0.0.0.0", self.api_host)

                audio_data = await self._download_audio(audio_url)
                if not audio_data:
                    logger.error("[GPTSoVITSTTSLocal] 无法获取音频数据")
                    return

            # 保存为临时文件
            file_name = f"{uuid.uuid4()}.wav"
            file_path = self.temp_dir / file_name
            with open(file_path, "wb") as f:
                f.write(audio_data)

            logger.info(f"[GPTSoVITSTTSLocal] 音频已保存: {file_path} ({len(audio_data)} bytes)")

            # 提取服务端 outputs 中的原始 wav 路径，供后续清理
            # 只有旧版 audio_url 接口才需要清理服务端文件
            server_wav_path = None
            if self.outputs_path and audio_url:
                parsed_url = urlparse(audio_url)
                wav_filename = Path(parsed_url.path).name
                if wav_filename:
                    server_wav_path = Path(self.outputs_path) / wav_filename
                    logger.debug(f"[GPTSoVITSTTSLocal] 将在 5 秒后清理服务端文件: {server_wav_path}")

            # 修改消息链
            # 构建音频组件，并将 text 设为空字符串，
            # 防止下游插件遍历 chain 拼接 text 时因 None 导致 join 报错
            record_comp = Comp.Record(file=str(file_path))
            try:
                if getattr(record_comp, 'text', None) is None:
                    record_comp.text = ""
            except Exception:
                pass

            if plain_indices:
                # 从后往前删，避免索引偏移
                for i in sorted(plain_indices, reverse=True):
                    del result.chain[i]

                insert_pos = min(plain_indices[0], len(result.chain))

                if self.send_text_with_audio:
                    # 先插入文本再插入音频，保证文本组件在前
                    result.chain.insert(insert_pos, Comp.Plain(f"\n[STT]\n{full_text}"))
                    result.chain.insert(insert_pos + 1, record_comp)
                else:
                    result.chain.insert(insert_pos, record_comp)
            else:
                # 理论上不会进这里，因为前面判断了 full_text
                result.chain.append(record_comp)

            # 清理本地临时文件（60 秒后）
            asyncio.create_task(self._cleanup_later(file_path, delay=60))

            # 清理服务端 outputs 中的原始 wav（5 秒后，语音发送成功后）
            if server_wav_path:
                asyncio.create_task(self._cleanup_later(server_wav_path, delay=5))

        except Exception as e:
            logger.error(f"[GPTSoVITSTTSLocal] TTS 转换异常: {e}")

    async def _cleanup_later(self, path: Path, delay=60):
        await asyncio.sleep(delay)
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"[GPTSoVITSTTSLocal] 清理临时文件: {path}")
        except Exception as e:
            logger.warning(f"[GPTSoVITSTTSLocal] 清理文件失败: {e}")