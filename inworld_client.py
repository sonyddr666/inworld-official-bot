import base64
import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests


BASE_URL = "https://api.inworld.ai"

MODELS: Dict[str, str] = {
    "inworld-tts-1.5-max": "1.5 Max",
    "inworld-tts-1.5-mini": "1.5 Mini",
    "inworld-tts-1-max": "1 Max",
    "inworld-tts-1": "1 Standard",
}

LANGUAGE_LABELS: Dict[str, str] = {
    "ALL": "Todas",
    "AUTO": "Auto",
    "AR_SA": "Arabic",
    "DE_DE": "Deutsch",
    "EN_US": "English",
    "ES_ES": "Espanol",
    "FR_FR": "Francais",
    "HE_IL": "Hebrew",
    "HI_IN": "Hindi",
    "IT_IT": "Italiano",
    "JA_JP": "Japanese",
    "KO_KR": "Korean",
    "NL_NL": "Nederlands",
    "PL_PL": "Polski",
    "PT_BR": "Portugues BR",
    "RU_RU": "Russian",
    "ZH_CN": "Chinese",
}

AUDIO_ENCODINGS: Dict[str, Dict[str, Any]] = {
    "MP3": {"extension": ".mp3", "mime": "audio/mpeg", "playable": True},
    "OGG_OPUS": {"extension": ".ogg", "mime": "audio/ogg", "playable": True},
    "LINEAR16": {"extension": ".wav", "mime": "audio/wav", "playable": True},
    "MULAW": {"extension": ".wav", "mime": "audio/wav", "playable": False},
    "ALAW": {"extension": ".wav", "mime": "audio/wav", "playable": False},
}

TIMESTAMP_TYPES = {
    "OFF": "Sem timestamps",
    "WORD": "Palavra",
    "CHARACTER": "Caractere",
}

DEFAULT_USER_STATE: Dict[str, Any] = {
    "api_auth": "",
    "active_voice_id": "",
    "active_voice_name": "",
    "language_filter": "ALL",
    "voice_scope": "all",
    "model_id": "inworld-tts-1.5-max",
    "speaking_rate": 1.0,
    "temperature": 1.0,
    "audio_encoding": "MP3",
    "sample_rate_hertz": 24000,
    "apply_text_normalization": True,
    "timestamp_type": "OFF",
}


class InworldAPIError(RuntimeError):
    pass


@dataclass
class SynthesizedAudio:
    audio_bytes: bytes
    extension: str
    mime_type: str
    playable: bool
    timestamp_info: Optional[Dict[str, Any]] = None


def normalize_basic_credential(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("Credencial vazia.")

    if raw.lower().startswith("basic "):
        token = raw.split(" ", 1)[1].strip()
        if not token:
            raise ValueError("Credencial Basic vazia.")
        return f"Basic {token}"

    if ":" in raw:
        encoded = base64.b64encode(raw.encode("utf-8")).decode("ascii")
        return f"Basic {encoded}"

    return f"Basic {raw}"


def mask_auth_header(value: str) -> str:
    auth = (value or "").strip()
    if not auth:
        return "nao configurada"
    token = auth.split(" ", 1)[1] if " " in auth else auth
    if len(token) <= 8:
        return "Basic ****"
    return f"Basic {token[:4]}...{token[-4:]}"


def normalize_language_code(code: Optional[str]) -> str:
    value = (code or "").strip().upper()
    if not value:
        return "ALL"

    aliases = {
        "ALL": "ALL",
        "AUTO": "AUTO",
        "AR": "AR_SA",
        "AR_SA": "AR_SA",
        "DE": "DE_DE",
        "DE_DE": "DE_DE",
        "EN": "EN_US",
        "EN_US": "EN_US",
        "ES": "ES_ES",
        "ES_ES": "ES_ES",
        "FR": "FR_FR",
        "FR_FR": "FR_FR",
        "HE": "HE_IL",
        "HE_IL": "HE_IL",
        "HI": "HI_IN",
        "HI_IN": "HI_IN",
        "IT": "IT_IT",
        "IT_IT": "IT_IT",
        "JA": "JA_JP",
        "JA_JP": "JA_JP",
        "KO": "KO_KR",
        "KO_KR": "KO_KR",
        "NL": "NL_NL",
        "NL_NL": "NL_NL",
        "PL": "PL_PL",
        "PL_PL": "PL_PL",
        "PT": "PT_BR",
        "PT_BR": "PT_BR",
        "RU": "RU_RU",
        "RU_RU": "RU_RU",
        "ZH": "ZH_CN",
        "ZH_CN": "ZH_CN",
    }
    if value in aliases:
        return aliases[value]
    raise ValueError(f"Idioma nao suportado: {code}")


def get_default_user_state() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_USER_STATE)


def is_custom_voice(voice: Dict[str, Any]) -> bool:
    voice_id = voice.get("voiceId", "") or ""
    return "__" in voice_id


def sort_voices(voices: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        voices,
        key=lambda item: (
            0 if is_custom_voice(item) else 1,
            (item.get("langCode") or "").upper(),
            (item.get("displayName") or item.get("voiceId") or "").lower(),
        ),
    )


class InworldClient:
    def __init__(self, auth_header: str, timeout: int = 90) -> None:
        self.auth_header = normalize_basic_credential(auth_header)
        self.timeout = timeout
        self.session = requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        headers = {
            "Authorization": self.auth_header,
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        response = self.session.request(
            method=method,
            url=f"{BASE_URL}{path}",
            headers=headers,
            params=params,
            json=json_body,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            detail = ""
            try:
                payload = response.json()
                detail = payload.get("message") or payload.get("error") or str(payload)
            except Exception:
                detail = response.text.strip()

            if response.status_code == 401:
                raise InworldAPIError("API key invalida ou sem prefixo Basic.")
            if response.status_code == 403:
                raise InworldAPIError(
                    "A API key nao tem permissao suficiente para esta operacao."
                )
            if response.status_code == 404:
                raise InworldAPIError("Recurso nao encontrado na Inworld.")
            if response.status_code == 429:
                raise InworldAPIError(
                    "Rate limit atingido na Inworld. Tente novamente em instantes."
                )
            raise InworldAPIError(f"Erro {response.status_code} da Inworld: {detail}")

        if response.content:
            return response.json()
        return {}

    def list_voices(
        self,
        *,
        languages: Optional[List[str]] = None,
        scope: str = "all",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if languages:
            params["languages"] = languages

        payload = self._request("GET", "/voices/v1/voices", params=params)
        voices = payload.get("voices", [])

        if scope == "custom":
            voices = [voice for voice in voices if is_custom_voice(voice)]
        elif scope == "system":
            voices = [voice for voice in voices if not is_custom_voice(voice)]

        return sort_voices(voices)

    def synthesize(
        self,
        *,
        text: str,
        voice_id: str,
        settings: Dict[str, Any],
    ) -> SynthesizedAudio:
        encoding = settings.get("audio_encoding", "MP3")
        if encoding not in AUDIO_ENCODINGS:
            raise ValueError(f"Encoding nao suportado: {encoding}")

        payload: Dict[str, Any] = {
            "text": text,
            "voiceId": voice_id,
            "modelId": settings.get("model_id", DEFAULT_USER_STATE["model_id"]),
            "temperature": float(settings.get("temperature", 1.0)),
            "applyTextNormalization": bool(
                settings.get("apply_text_normalization", True)
            ),
            "audioConfig": {
                "audioEncoding": encoding,
                "sampleRateHertz": int(
                    settings.get("sample_rate_hertz", DEFAULT_USER_STATE["sample_rate_hertz"])
                ),
            },
        }

        speaking_rate = settings.get("speaking_rate")
        if speaking_rate not in (None, "", 1, 1.0):
            payload["audioConfig"]["speakingRate"] = float(speaking_rate)

        timestamp_type = settings.get("timestamp_type", "OFF")
        if timestamp_type and timestamp_type != "OFF":
            payload["timestampType"] = timestamp_type

        try:
            data = self._request("POST", "/tts/v1/voice", json_body=payload)
        except InworldAPIError as exc:
            if "speakingRate" in payload["audioConfig"] and "400" in str(exc):
                payload["audioConfig"].pop("speakingRate", None)
                data = self._request("POST", "/tts/v1/voice", json_body=payload)
            else:
                raise
        audio_content = data.get("audioContent")
        if not audio_content:
            raise InworldAPIError("A resposta nao trouxe audioContent.")

        meta = AUDIO_ENCODINGS[encoding]
        return SynthesizedAudio(
            audio_bytes=base64.b64decode(audio_content),
            extension=meta["extension"],
            mime_type=meta["mime"],
            playable=bool(meta["playable"]),
            timestamp_info=data.get("timestampInfo"),
        )

    def clone_voice(
        self,
        *,
        display_name: str,
        lang_code: str,
        voice_samples: List[Dict[str, str]],
        description: str = "",
        tags: Optional[List[str]] = None,
        remove_background_noise: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "displayName": display_name,
            "langCode": normalize_language_code(lang_code),
            "voiceSamples": voice_samples,
            "audioProcessingConfig": {
                "removeBackgroundNoise": bool(remove_background_noise)
            },
        }

        if description:
            payload["description"] = description
        if tags:
            payload["tags"] = tags

        data = self._request("POST", "/voices/v1/voices:clone", json_body=payload)
        voice = data.get("voice")
        if not voice:
            raise InworldAPIError("A resposta nao trouxe a voz clonada.")
        return voice

    def delete_voice(self, voice_id: str) -> None:
        self._request("DELETE", f"/voices/v1/voices/{voice_id}")
