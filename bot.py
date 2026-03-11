import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
)

from inworld_client import (
    AUDIO_ENCODINGS,
    LANGUAGE_LABELS,
    MODELS,
    InworldAPIError,
    InworldClient,
    is_custom_voice,
    mask_auth_header,
    normalize_basic_credential,
    normalize_language_code,
)
from storage import LocalStateStore


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TMP_CLONE_DIR = DATA_DIR / "clone_uploads"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TMP_CLONE_DIR.mkdir(exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
DEFAULT_AUTH_HEADER = os.getenv("INWORLD_BASE64_CREDENTIAL", "").strip()
DEFAULT_VOICE_ID = os.getenv("INWORLD_DEFAULT_VOICE_ID", "").strip()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_CHARS_PER_REQUEST = int(os.getenv("INWORLD_MAX_CHARS_PER_REQUEST", "1900"))
MAX_CLONE_SAMPLES = 3
MAX_CLONE_TOTAL_SIZE = 16 * 1024 * 1024
SAMPLE_RATE_OPTIONS = [16000, 22050, 24000, 44100, 48000]

STORE = LocalStateStore(DATA_DIR / "user_state.json")
API_SEMAPHORE = asyncio.Semaphore(3)

CLONE_NAME, CLONE_LANGUAGE, CLONE_DESCRIPTION, CLONE_TAGS, CLONE_NOISE, CLONE_SAMPLES = range(6)

LANGUAGE_ORDER = [
    "ALL",
    "PT_BR",
    "EN_US",
    "ES_ES",
    "FR_FR",
    "DE_DE",
    "IT_IT",
    "JA_JP",
    "KO_KR",
    "ZH_CN",
    "RU_RU",
    "AR_SA",
    "HI_IN",
    "HE_IL",
    "PL_PL",
    "NL_NL",
]

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return cleaned[:60] or "audio"


def build_client_from_state(state: Dict[str, Any]) -> InworldClient:
    auth_header = state.get("api_auth") or DEFAULT_AUTH_HEADER
    if not auth_header:
        raise InworldAPIError(
            "Nenhuma API key configurada. Use /apikey Basic_BASE64 ou /apikey key_id:key_secret."
        )
    return InworldClient(auth_header)


def get_state(user_id: int) -> Dict[str, Any]:
    state = STORE.get_user_state(user_id)
    if DEFAULT_VOICE_ID and not state.get("active_voice_id"):
        state["active_voice_id"] = DEFAULT_VOICE_ID
        state["active_voice_name"] = DEFAULT_VOICE_ID
    return state


def save_state(user_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
    return STORE.save_user_state(user_id, state)


def format_settings(state: Dict[str, Any]) -> str:
    voice_name = state.get("active_voice_name") or state.get("active_voice_id") or "nao definida"
    return (
        "Configuracao atual\n"
        f"- API key: {mask_auth_header(state.get('api_auth') or DEFAULT_AUTH_HEADER)}\n"
        f"- Voz: {voice_name}\n"
        f"- Filtro de idioma: {LANGUAGE_LABELS.get(state.get('language_filter', 'ALL'), state.get('language_filter', 'ALL'))}\n"
        f"- Escopo de vozes: {state.get('voice_scope', 'all')}\n"
        f"- Modelo: {state.get('model_id')}\n"
        f"- Velocidade: {state.get('speaking_rate')}\n"
        f"- Temperatura: {state.get('temperature')}\n"
        f"- Encoding: {state.get('audio_encoding')}\n"
        f"- Sample rate: {state.get('sample_rate_hertz')} Hz\n"
        f"- Normalizacao: {'on' if state.get('apply_text_normalization') else 'off'}\n"
        f"- Timestamps: {state.get('timestamp_type')}"
    )


def chunk_text(text: str, limit: int = MAX_CHARS_PER_REQUEST) -> List[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []
    if len(normalized) <= limit:
        return [normalized]

    chunks: List[str] = []
    paragraphs = [part.strip() for part in normalized.split("\n") if part.strip()]
    current = ""

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > limit:
                words = sentence.split()
                piece = ""
                for word in words:
                    if len(word) > limit:
                        if piece:
                            chunks.append(piece.strip())
                            piece = ""
                        for start in range(0, len(word), limit):
                            chunks.append(word[start : start + limit])
                        continue
                    candidate = f"{piece} {word}".strip()
                    if len(candidate) <= limit:
                        piece = candidate
                    else:
                        chunks.append(piece.strip())
                        piece = word
                if piece:
                    chunks.append(piece.strip())
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= limit:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = sentence

    if current:
        chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def parse_tags(raw: str) -> List[str]:
    tags: List[str] = []
    for item in raw.split(","):
        cleaned = item.strip().replace(" ", "_")
        if cleaned:
            tags.append(cleaned[:32])
    return tags[:12]


def parse_float_arg(value: str, minimum: float, maximum: float, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{label} invalido.") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"{label} deve ficar entre {minimum} e {maximum}.")
    return round(result, 2)


def voice_scope_label(scope: str) -> str:
    return {
        "all": "todas",
        "system": "somente sistema",
        "custom": "somente clonadas",
    }.get(scope, scope)


def get_language_rows(include_all: bool = True, prefix: str = "langpick") -> List[List[InlineKeyboardButton]]:
    codes = [code for code in LANGUAGE_ORDER if include_all or code != "ALL"]
    rows: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for code in codes:
        row.append(
            InlineKeyboardButton(LANGUAGE_LABELS.get(code, code), callback_data=f"{prefix}:{code}")
        )
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return rows


async def reply_no_key(update: Update) -> None:
    await update.effective_message.reply_text(
        "Nenhuma API key oficial configurada.\n"
        "Use /apikey Basic_BASE64 ou /apikey key_id:key_secret.\n"
        "Para remover a chave salva: /apikey clear"
    )


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Bot Telegram Inworld API oficial\n\n"
        "Comandos principais:\n"
        "/apikey - salvar ou limpar API key\n"
        "/settings - ver configuracao atual\n"
        "/voices - listar e trocar voz\n"
        "/myvoices - listar clones e deletar\n"
        "/clone - iniciar clone de voz\n"
        "/model - trocar modelo\n"
        "/speed - ajustar velocidade\n"
        "/temp - ajustar temperatura\n"
        "/encoding - trocar formato de audio\n"
        "/samplerate - trocar sample rate\n"
        "/normalize - ligar/desligar normalizacao\n"
        "/timestamps - exportar alinhamento\n\n"
        "Envie texto ou um .txt para receber o audio."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start_command(update, context)


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(update.effective_user.id)
    await update.message.reply_text(format_settings(state))


async def apikey_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)
    if not context.args:
        await update.message.reply_text(
            "Uso:\n"
            "/apikey Basic SEU_BASE64\n"
            "/apikey key_id:key_secret\n"
            "/apikey clear\n\n"
            f"Atual: {mask_auth_header(state.get('api_auth') or DEFAULT_AUTH_HEADER)}"
        )
        return

    raw = " ".join(context.args).strip()
    if raw.lower() == "clear":
        state["api_auth"] = ""
        save_state(user_id, state)
        await update.message.reply_text("API key removida do armazenamento local do bot.")
        return

    try:
        normalized = normalize_basic_credential(raw)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return

    state["api_auth"] = normalized
    save_state(user_id, state)
    await update.message.reply_text(
        f"API key salva com sucesso.\nMascarada: {mask_auth_header(normalized)}"
    )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)

    if context.args:
        model_id = context.args[0].strip()
        if model_id not in MODELS:
            await update.message.reply_text(f"Modelo invalido. Opcoes: {', '.join(MODELS)}")
            return
        state["model_id"] = model_id
        save_state(user_id, state)
        await update.message.reply_text(f"Modelo definido para {model_id}.")
        return

    rows = [
        [InlineKeyboardButton(f"{'OK ' if key == state['model_id'] else ''}{label}", callback_data=f"set:model:{key}")]
        for key, label in MODELS.items()
    ]
    await update.message.reply_text(
        "Escolha o modelo:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def speed_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)
    if context.args:
        try:
            value = parse_float_arg(context.args[0], 0.25, 4.0, "Velocidade")
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return
        state["speaking_rate"] = value
        save_state(user_id, state)
        await update.message.reply_text(f"Velocidade definida para {value}.")
        return

    buttons = [
        [
            InlineKeyboardButton("0.75x", callback_data="set:speed:0.75"),
            InlineKeyboardButton("1.0x", callback_data="set:speed:1.0"),
            InlineKeyboardButton("1.25x", callback_data="set:speed:1.25"),
        ],
        [
            InlineKeyboardButton("1.5x", callback_data="set:speed:1.5"),
            InlineKeyboardButton("2.0x", callback_data="set:speed:2.0"),
            InlineKeyboardButton("3.0x", callback_data="set:speed:3.0"),
        ],
    ]
    await update.message.reply_text(
        "Escolha a velocidade ou use /speed 1.15",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def temperature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)
    if context.args:
        try:
            value = parse_float_arg(context.args[0], 0.7, 1.5, "Temperatura")
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return
        state["temperature"] = value
        save_state(user_id, state)
        await update.message.reply_text(f"Temperatura definida para {value}.")
        return

    buttons = [
        [
            InlineKeyboardButton("0.7", callback_data="set:temp:0.7"),
            InlineKeyboardButton("1.0", callback_data="set:temp:1.0"),
            InlineKeyboardButton("1.2", callback_data="set:temp:1.2"),
        ],
        [
            InlineKeyboardButton("1.35", callback_data="set:temp:1.35"),
            InlineKeyboardButton("1.5", callback_data="set:temp:1.5"),
        ],
    ]
    await update.message.reply_text(
        "Escolha a temperatura ou use /temp 1.1",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def encoding_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)

    if context.args:
        value = context.args[0].strip().upper()
        if value not in AUDIO_ENCODINGS:
            await update.message.reply_text(
                f"Encoding invalido. Opcoes: {', '.join(AUDIO_ENCODINGS)}"
            )
            return
        state["audio_encoding"] = value
        save_state(user_id, state)
        await update.message.reply_text(f"Encoding definido para {value}.")
        return

    rows = [
        [
            InlineKeyboardButton(
                f"{'OK ' if key == state['audio_encoding'] else ''}{key}",
                callback_data=f"set:encoding:{key}",
            )
        ]
        for key in AUDIO_ENCODINGS
    ]
    await update.message.reply_text(
        "Escolha o formato do audio:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def samplerate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    state = get_state(user_id)

    if context.args:
        try:
            value = int(context.args[0])
        except ValueError:
            await update.message.reply_text("Sample rate invalido.")
            return
        state["sample_rate_hertz"] = value
        save_state(user_id, state)
        await update.message.reply_text(f"Sample rate definido para {value} Hz.")
        return

    rows = [
        [
            InlineKeyboardButton(
                f"{'OK ' if value == state['sample_rate_hertz'] else ''}{value} Hz",
                callback_data=f"set:samplerate:{value}",
            )
        ]
        for value in SAMPLE_RATE_OPTIONS
    ]
    await update.message.reply_text(
        "Escolha o sample rate:",
        reply_markup=InlineKeyboardMarkup(rows),
    )


async def normalize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    buttons = [
        [
            InlineKeyboardButton("Normalizacao ON", callback_data="set:normalize:on"),
            InlineKeyboardButton("Normalizacao OFF", callback_data="set:normalize:off"),
        ]
    ]
    await update.message.reply_text(
        "Normalizacao ajuda em abreviacoes, datas e numeros.",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def timestamps_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = [
        [
            InlineKeyboardButton("OFF", callback_data="set:timestamps:OFF"),
            InlineKeyboardButton("WORD", callback_data="set:timestamps:WORD"),
            InlineKeyboardButton("CHARACTER", callback_data="set:timestamps:CHARACTER"),
        ]
    ]
    await update.message.reply_text(
        "Timestamps aumentam a latencia e hoje funcionam melhor em ingles.",
        reply_markup=InlineKeyboardMarkup(rows),
    )


def build_voice_browser_keyboard(
    voices: List[Dict[str, Any]],
    *,
    page: int,
    page_size: int,
    language_filter: str,
    scope: str,
    allow_delete: bool = False,
) -> InlineKeyboardMarkup:
    start = page * page_size
    end = start + page_size
    page_items = voices[start:end]
    rows: List[List[InlineKeyboardButton]] = []

    for index, voice in enumerate(page_items):
        absolute_index = start + index
        title = (voice.get("displayName") or voice.get("voiceId") or "?")[:22]
        icon = "IVC" if is_custom_voice(voice) else "SYS"
        rows.append(
            [
                InlineKeyboardButton(
                    f"{icon} {title}",
                    callback_data=f"pickvoice:{language_filter}:{scope}:{page}:{absolute_index}",
                )
            ]
        )
        if allow_delete and is_custom_voice(voice):
            rows.append(
                [
                    InlineKeyboardButton(
                        "Apagar",
                        callback_data=f"confirmdelete:{language_filter}:{scope}:{page}:{absolute_index}",
                    )
                ]
            )

    nav: List[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton(
                "Prev", callback_data=f"browse:{language_filter}:{scope}:{page - 1}:{int(allow_delete)}"
            )
        )
    nav.append(
        InlineKeyboardButton(
            f"Idioma {LANGUAGE_LABELS.get(language_filter, language_filter)}",
            callback_data=f"picklangbrowse:{scope}:{int(allow_delete)}",
        )
    )
    if end < len(voices):
        nav.append(
            InlineKeyboardButton(
                "Next", callback_data=f"browse:{language_filter}:{scope}:{page + 1}:{int(allow_delete)}"
            )
        )
    rows.append(nav)

    rows.append(
        [
            InlineKeyboardButton(
                "Todas", callback_data=f"browse:ALL:{scope}:0:{int(allow_delete)}"
            ),
            InlineKeyboardButton(
                "Sistema", callback_data=f"browse:{language_filter}:system:0:{int(allow_delete)}"
            ),
            InlineKeyboardButton(
                "Clonadas", callback_data=f"browse:{language_filter}:custom:0:{int(allow_delete)}"
            ),
        ]
    )
    return InlineKeyboardMarkup(rows)


async def show_voice_browser(
    message,
    *,
    state: Dict[str, Any],
    user_id: int,
    page: int = 0,
    allow_delete: bool = False,
    edit: bool = False,
) -> None:
    try:
        client = build_client_from_state(state)
    except InworldAPIError:
        text = "Nenhuma API key configurada. Use /apikey antes de navegar nas vozes."
        if edit:
            await message.edit_text(text)
        else:
            await message.reply_text(text)
        return

    language_filter = state.get("language_filter", "ALL")
    scope = state.get("voice_scope", "all")
    languages = None if language_filter == "ALL" else [normalize_language_code(language_filter)]

    async with API_SEMAPHORE:
        voices = await asyncio.to_thread(client.list_voices, languages=languages, scope=scope)

    if not voices:
        text = "Nenhuma voz encontrada para o filtro atual."
        if edit:
            await message.edit_text(text)
        else:
            await message.reply_text(text)
        return

    page_size = 8
    max_page = max((len(voices) - 1) // page_size, 0)
    page = max(0, min(page, max_page))
    start = page * page_size
    end = min(start + page_size, len(voices))
    items = voices[start:end]

    lines = [
        "Lista de vozes",
        f"- Idioma: {LANGUAGE_LABELS.get(language_filter, language_filter)}",
        f"- Escopo: {voice_scope_label(scope)}",
        f"- Pagina: {page + 1}/{max_page + 1}",
        "",
    ]
    for voice in items:
        voice_id = voice.get("voiceId", "?")
        display_name = voice.get("displayName") or voice_id
        lang_code = voice.get("langCode", "?")
        tag_preview = ", ".join(voice.get("tags", [])[:3]) or "-"
        prefix = "IVC" if is_custom_voice(voice) else "SYS"
        lines.append(f"[{prefix}] {display_name} | {lang_code} | {tag_preview}")

    markup = build_voice_browser_keyboard(
        voices,
        page=page,
        page_size=page_size,
        language_filter=language_filter,
        scope=scope,
        allow_delete=allow_delete,
    )

    if edit:
        await message.edit_text("\n".join(lines), reply_markup=markup)
    else:
        await message.reply_text("\n".join(lines), reply_markup=markup)

    save_state(user_id, state)


async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(update.effective_user.id)
    if context.args:
        try:
            state["language_filter"] = normalize_language_code(context.args[0])
        except ValueError as exc:
            await update.message.reply_text(str(exc))
            return
    state["voice_scope"] = "all"
    save_state(update.effective_user.id, state)
    await show_voice_browser(update.message, state=state, user_id=update.effective_user.id)


async def myvoices_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(update.effective_user.id)
    state["voice_scope"] = "custom"
    save_state(update.effective_user.id, state)
    await show_voice_browser(
        update.message,
        state=state,
        user_id=update.effective_user.id,
        allow_delete=True,
    )


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await voices_command(update, context)


async def set_voice_by_index(
    query,
    *,
    user_id: int,
    language_filter: str,
    scope: str,
    index: int,
) -> None:
    state = get_state(user_id)
    state["language_filter"] = language_filter
    state["voice_scope"] = scope

    try:
        client = build_client_from_state(state)
    except InworldAPIError:
        await query.edit_message_text(
            "Nenhuma API key configurada. Use /apikey antes de navegar nas vozes."
        )
        return

    languages = None if language_filter == "ALL" else [normalize_language_code(language_filter)]
    async with API_SEMAPHORE:
        voices = await asyncio.to_thread(client.list_voices, languages=languages, scope=scope)
    if index >= len(voices):
        await query.answer("Lista expirada. Abra /voices novamente.", show_alert=True)
        return

    selected = voices[index]
    state["active_voice_id"] = selected.get("voiceId", "")
    state["active_voice_name"] = selected.get("displayName") or state["active_voice_id"]
    save_state(user_id, state)
    await query.edit_message_text(
        "Voz alterada\n"
        f"- Nome: {state['active_voice_name']}\n"
        f"- ID: {state['active_voice_id']}\n"
        f"- Idioma: {selected.get('langCode', '-')}"
    )


async def clone_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    state = get_state(update.effective_user.id)
    if not (state.get("api_auth") or DEFAULT_AUTH_HEADER):
        await reply_no_key(update)
        return ConversationHandler.END

    clone_dir = TMP_CLONE_DIR / str(update.effective_user.id)
    clone_dir.mkdir(parents=True, exist_ok=True)
    context.user_data["clone"] = {
        "display_name": "",
        "lang_code": "PT_BR",
        "description": "",
        "tags": [],
        "remove_background_noise": True,
        "samples": [],
        "clone_dir": str(clone_dir),
    }
    await update.message.reply_text(
        "Clone de voz iniciado.\nEnvie o nome exibido da voz."
    )
    return CLONE_NAME


async def clone_receive_name(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END

    clone["display_name"] = update.message.text.strip()[:60]
    rows = get_language_rows(include_all=False)
    await update.message.reply_text(
        "Escolha o idioma principal da voz clonada:",
        reply_markup=InlineKeyboardMarkup(rows),
    )
    return CLONE_LANGUAGE


async def clone_pick_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    lang_code = query.data.split(":", 1)[1]
    clone["lang_code"] = lang_code
    await query.edit_message_text(
        f"Idioma definido para {LANGUAGE_LABELS.get(lang_code, lang_code)}.\n"
        "Agora envie uma descricao curta da voz ou /skip para pular."
    )
    return CLONE_DESCRIPTION


async def clone_receive_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    clone["description"] = update.message.text.strip()[:240]
    await update.message.reply_text(
        "Envie tags separadas por virgula ou /skip para pular.\n"
        "Exemplo: masculina, calorosa, narracao"
    )
    return CLONE_TAGS


async def clone_skip_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    clone["description"] = ""
    await update.message.reply_text(
        "Descricao pulada.\nEnvie tags separadas por virgula ou /skip para pular."
    )
    return CLONE_TAGS


async def clone_receive_tags(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    clone["tags"] = parse_tags(update.message.text.strip())
    buttons = [
        [
            InlineKeyboardButton("Ruido ON", callback_data="clonenoise:on"),
            InlineKeyboardButton("Ruido OFF", callback_data="clonenoise:off"),
        ]
    ]
    await update.message.reply_text(
        "Remover ruido de fundo?",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return CLONE_NOISE


async def clone_skip_tags(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    clone["tags"] = []
    buttons = [
        [
            InlineKeyboardButton("Ruido ON", callback_data="clonenoise:on"),
            InlineKeyboardButton("Ruido OFF", callback_data="clonenoise:off"),
        ]
    ]
    await update.message.reply_text(
        "Sem tags.\nRemover ruido de fundo?",
        reply_markup=InlineKeyboardMarkup(buttons),
    )
    return CLONE_NOISE


async def clone_pick_noise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END
    clone["remove_background_noise"] = query.data.endswith(":on")
    await query.edit_message_text(
        "Agora envie de 1 a 3 amostras de audio.\n"
        "Formatos recomendados pela Inworld: wav, mp3 ou webm.\n"
        "Cada amostra deve ter idealmente 5 a 15 segundos.\n"
        "Se quiser, coloque a transcricao no caption.\n"
        "Quando terminar, envie /clone_done."
    )
    return CLONE_SAMPLES


async def clone_expect_language_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text("Escolha o idioma usando os botoes.")
    return CLONE_LANGUAGE


async def clone_expect_noise_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text("Escolha a opcao de ruido usando os botoes.")
    return CLONE_NOISE


async def clone_expect_samples(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.effective_message.reply_text(
        "Nesta etapa envie audio(s) ou finalize com /clone_done."
    )
    return CLONE_SAMPLES


async def clone_receive_sample(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        return ConversationHandler.END

    samples = clone.get("samples", [])
    if len(samples) >= MAX_CLONE_SAMPLES:
        await update.message.reply_text("Limite de 3 amostras atingido. Envie /clone_done.")
        return CLONE_SAMPLES

    message = update.message
    tg_file = None
    file_size = 0
    duration = None
    file_name = None

    if message.audio:
        tg_file = await message.audio.get_file()
        file_size = message.audio.file_size or 0
        duration = message.audio.duration
        file_name = message.audio.file_name or "sample.mp3"
    elif message.voice:
        tg_file = await message.voice.get_file()
        file_size = message.voice.file_size or 0
        duration = message.voice.duration
        file_name = "voice_note.ogg"
    elif message.document:
        tg_file = await message.document.get_file()
        file_size = message.document.file_size or 0
        file_name = message.document.file_name or "sample.bin"
        if Path(file_name).suffix.lower() not in {".wav", ".mp3", ".webm", ".ogg", ".oga"}:
            await update.message.reply_text(
                "Para clone, envie wav, mp3, webm ou uma nota de voz."
            )
            return CLONE_SAMPLES
    else:
        await update.message.reply_text("Envie audio, voice note ou documento de audio.")
        return CLONE_SAMPLES

    total_size = sum(sample.get("size", 0) for sample in samples) + file_size
    if total_size > MAX_CLONE_TOTAL_SIZE:
        await update.message.reply_text("As amostras passaram de 16 MB no total.")
        return CLONE_SAMPLES

    safe_name = sanitize_filename(file_name)
    dest = Path(clone["clone_dir"]) / f"{len(samples) + 1}_{safe_name}"
    await tg_file.download_to_drive(custom_path=str(dest))

    transcription = (message.caption or "").strip()
    sample = {
        "path": str(dest),
        "size": file_size,
        "duration": duration,
        "transcription": transcription,
    }
    samples.append(sample)
    clone["samples"] = samples

    warning = ""
    if duration and (duration < 5 or duration > 15):
        warning = "\nAviso: a Inworld recomenda 5 a 15 segundos por amostra."
    if dest.suffix.lower() not in {".wav", ".mp3", ".webm", ".ogg", ".oga"}:
        warning += "\nAviso: formato fora do recomendado pode falhar."

    await update.message.reply_text(
        f"Amostra {len(samples)}/{MAX_CLONE_SAMPLES} salva.{warning}\n"
        "Envie mais uma ou /clone_done."
    )
    return CLONE_SAMPLES


async def cleanup_clone_uploads(clone: Dict[str, Any]) -> None:
    clone_dir = Path(clone.get("clone_dir", ""))
    if not clone_dir.exists():
        return
    for path in clone_dir.glob("*"):
        try:
            path.unlink()
        except OSError:
            pass
    try:
        clone_dir.rmdir()
    except OSError:
        pass


async def clone_finish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.get("clone")
    if not clone:
        await update.message.reply_text("Nenhum clone em andamento.")
        return ConversationHandler.END

    if not clone.get("samples"):
        await update.message.reply_text("Envie ao menos uma amostra antes de /clone_done.")
        return CLONE_SAMPLES

    user_id = update.effective_user.id
    state = get_state(user_id)
    try:
        client = build_client_from_state(state)
    except InworldAPIError:
        await reply_no_key(update)
        return ConversationHandler.END

    voice_samples = []
    for sample in clone["samples"]:
        audio_bytes = Path(sample["path"]).read_bytes()
        payload = {"audioData": base64.b64encode(audio_bytes).decode("ascii")}
        if sample.get("transcription"):
            payload["transcription"] = sample["transcription"]
        voice_samples.append(payload)

    status = await update.message.reply_text("Clonando voz na Inworld...")
    try:
        async with API_SEMAPHORE:
            voice = await asyncio.to_thread(
                client.clone_voice,
                display_name=clone["display_name"],
                lang_code=clone["lang_code"],
                voice_samples=voice_samples,
                description=clone.get("description", ""),
                tags=clone.get("tags", []),
                remove_background_noise=clone.get("remove_background_noise", True),
            )
    except InworldAPIError as exc:
        await status.edit_text(f"Falha ao clonar voz: {exc}")
        await cleanup_clone_uploads(clone)
        context.user_data.pop("clone", None)
        return ConversationHandler.END

    state["active_voice_id"] = voice.get("voiceId", "")
    state["active_voice_name"] = voice.get("displayName") or state["active_voice_id"]
    save_state(user_id, state)
    await status.edit_text(
        "Clone criado com sucesso.\n"
        f"- Nome: {voice.get('displayName')}\n"
        f"- ID: {voice.get('voiceId')}\n"
        f"- Idioma: {voice.get('langCode')}\n"
        "Essa voz ja ficou ativa para suas proximas sinteses."
    )
    await cleanup_clone_uploads(clone)
    context.user_data.pop("clone", None)
    return ConversationHandler.END


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    clone = context.user_data.pop("clone", None)
    if clone:
        await cleanup_clone_uploads(clone)
    await update.effective_message.reply_text("Operacao cancelada.")
    return ConversationHandler.END


async def read_text_from_document(update: Update) -> Optional[str]:
    document = update.message.document
    if not document:
        return None
    if document.file_size and document.file_size > 1024 * 1024:
        raise ValueError("Arquivo de texto maior que 1 MB.")
    tg_file = await document.get_file()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_path = Path(temp_file.name)
    await tg_file.download_to_drive(custom_path=str(temp_path))
    raw = temp_path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


async def synthesize_parts(
    update: Update,
    *,
    text: str,
    state: Dict[str, Any],
) -> None:
    if not text.strip():
        return

    if not state.get("active_voice_id"):
        await update.effective_message.reply_text(
            "Nenhuma voz ativa. Use /voices para escolher uma voz."
        )
        return

    try:
        client = build_client_from_state(state)
    except InworldAPIError:
        await reply_no_key(update)
        return

    chunks = chunk_text(text)
    if not chunks:
        await update.effective_message.reply_text("Nenhum texto valido encontrado.")
        return

    total_chars = len(text)
    warning_lines = []
    if total_chars > MAX_CHARS_PER_REQUEST:
        warning_lines.append(
            f"Texto longo detectado: {total_chars} caracteres. Vou dividir em {len(chunks)} parte(s)."
        )
    if len(chunks) > 8:
        warning_lines.append("Aviso: muitas partes vao aumentar a latencia e o custo.")
    if state.get("audio_encoding") != "MP3":
        warning_lines.append(
            f"Encoding atual: {state.get('audio_encoding')}. MP3 segue disponivel em /encoding."
        )
    if warning_lines:
        await update.effective_message.reply_text("\n".join(warning_lines))

    progress = await update.effective_message.reply_text(
        f"Gerando {len(chunks)} audio(s) com a voz {state.get('active_voice_name') or state.get('active_voice_id')}..."
    )

    timestamp_enabled = state.get("timestamp_type", "OFF") != "OFF"
    for idx, chunk in enumerate(chunks, start=1):
        await update.effective_chat.send_action(ChatAction.UPLOAD_AUDIO)
        async with API_SEMAPHORE:
            try:
                result = await asyncio.to_thread(
                    client.synthesize,
                    text=chunk,
                    voice_id=state["active_voice_id"],
                    settings=state,
                )
            except InworldAPIError as exc:
                await progress.edit_text(f"Falha na parte {idx}/{len(chunks)}: {exc}")
                return

        base_name = sanitize_filename(
            f"{update.effective_user.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_part{idx}"
        )
        audio_path = OUTPUT_DIR / f"{base_name}{result.extension}"
        audio_path.write_bytes(result.audio_bytes)

        caption = (
            f"Parte {idx}/{len(chunks)}\n"
            f"Modelo: {state.get('model_id')}\n"
            f"Velocidade: {state.get('speaking_rate')} | Temp: {state.get('temperature')}"
        )

        try:
            with audio_path.open("rb") as audio_stream:
                if result.playable:
                    await update.effective_message.reply_audio(
                        audio=audio_stream,
                        filename=audio_path.name,
                        caption=caption,
                        title=audio_path.stem,
                    )
                else:
                    await update.effective_message.reply_document(
                        document=audio_stream,
                        filename=audio_path.name,
                        caption=caption,
                    )

            if timestamp_enabled and result.timestamp_info:
                json_path = OUTPUT_DIR / f"{base_name}.json"
                json_path.write_text(
                    json.dumps(result.timestamp_info, ensure_ascii=True, indent=2),
                    encoding="utf-8",
                )
                with json_path.open("rb") as json_stream:
                    await update.effective_message.reply_document(
                        document=json_stream,
                        filename=json_path.name,
                        caption=f"Timestamps da parte {idx}/{len(chunks)}",
                    )
                try:
                    json_path.unlink()
                except OSError:
                    pass
        finally:
            try:
                audio_path.unlink()
            except OSError:
                pass

        await progress.edit_text(f"Progresso: {idx}/{len(chunks)} parte(s) concluida(s).")

    await progress.edit_text(f"Concluido. {len(chunks)} audio(s) enviados.")


async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(update.effective_user.id)
    await synthesize_parts(update, text=update.message.text, state=state)


async def document_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    document = update.message.document
    if not document:
        return
    file_name = (document.file_name or "").lower()
    if not any(file_name.endswith(ext) for ext in (".txt", ".md")):
        await update.message.reply_text("Envie um .txt ou .md para TTS.")
        return
    try:
        text = await read_text_from_document(update)
    except ValueError as exc:
        await update.message.reply_text(str(exc))
        return
    state = get_state(update.effective_user.id)
    await synthesize_parts(update, text=text or "", state=state)


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    state = get_state(user_id)
    parts = query.data.split(":")
    action = parts[0]

    if action == "set":
        kind = parts[1]
        value = parts[2]
        if kind == "model":
            state["model_id"] = value
        elif kind == "speed":
            state["speaking_rate"] = float(value)
        elif kind == "temp":
            state["temperature"] = float(value)
        elif kind == "encoding":
            state["audio_encoding"] = value
        elif kind == "samplerate":
            state["sample_rate_hertz"] = int(value)
        elif kind == "normalize":
            state["apply_text_normalization"] = value == "on"
        elif kind == "timestamps":
            state["timestamp_type"] = value
        save_state(user_id, state)
        await query.edit_message_text(format_settings(state))
        return

    if action == "browse":
        language_filter = parts[1]
        scope = parts[2]
        page = int(parts[3])
        allow_delete = bool(int(parts[4]))
        state["language_filter"] = language_filter
        state["voice_scope"] = scope
        save_state(user_id, state)
        await show_voice_browser(
            query.message,
            state=state,
            user_id=user_id,
            page=page,
            allow_delete=allow_delete,
            edit=True,
        )
        return

    if action == "picklangbrowse":
        scope = parts[1]
        allow_delete = int(parts[2])
        rows = get_language_rows(True, prefix=f"browsepick:{scope}:{allow_delete}")
        await query.edit_message_text(
            "Escolha o idioma para filtrar:",
            reply_markup=InlineKeyboardMarkup(rows),
        )
        return

    if action == "browsepick":
        scope = parts[1]
        allow_delete = int(parts[2])
        language_filter = parts[3]
        state["language_filter"] = language_filter
        state["voice_scope"] = scope
        save_state(user_id, state)
        await show_voice_browser(
            query.message,
            state=state,
            user_id=user_id,
            page=0,
            allow_delete=bool(allow_delete),
            edit=True,
        )
        return

    if action == "pickvoice":
        language_filter = parts[1]
        scope = parts[2]
        absolute_index = int(parts[4])
        await set_voice_by_index(
            query,
            user_id=user_id,
            language_filter=language_filter,
            scope=scope,
            index=absolute_index,
        )
        return

    if action == "confirmdelete":
        language_filter = parts[1]
        scope = parts[2]
        page = int(parts[3])
        absolute_index = int(parts[4])
        buttons = [
            [
                InlineKeyboardButton(
                    "Confirmar",
                    callback_data=f"deletevoice:{language_filter}:{scope}:{page}:{absolute_index}",
                ),
                InlineKeyboardButton(
                    "Voltar",
                    callback_data=f"browse:{language_filter}:{scope}:{page}:1",
                ),
            ]
        ]
        await query.edit_message_text(
            "Apagar esta voz clonada?",
            reply_markup=InlineKeyboardMarkup(buttons),
        )
        return

    if action == "deletevoice":
        language_filter = parts[1]
        scope = parts[2]
        absolute_index = int(parts[4])
        try:
            client = build_client_from_state(state)
        except InworldAPIError:
            await query.edit_message_text("Configure /apikey antes de apagar vozes.")
            return
        languages = None if language_filter == "ALL" else [normalize_language_code(language_filter)]
        async with API_SEMAPHORE:
            voices = await asyncio.to_thread(client.list_voices, languages=languages, scope=scope)
        if absolute_index >= len(voices):
            await query.edit_message_text("Lista expirada. Abra /myvoices novamente.")
            return
        voice = voices[absolute_index]
        if not is_custom_voice(voice):
            await query.edit_message_text("Somente vozes clonadas podem ser apagadas.")
            return
        try:
            async with API_SEMAPHORE:
                await asyncio.to_thread(client.delete_voice, voice.get("voiceId", ""))
        except InworldAPIError as exc:
            await query.edit_message_text(f"Falha ao apagar voz: {exc}")
            return

        if state.get("active_voice_id") == voice.get("voiceId"):
            state["active_voice_id"] = DEFAULT_VOICE_ID
            state["active_voice_name"] = DEFAULT_VOICE_ID
            save_state(user_id, state)

        await query.edit_message_text(
            f"Voz apagada: {voice.get('displayName') or voice.get('voiceId')}"
        )
        return


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Defina TELEGRAM_BOT_TOKEN no .env.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    clone_handler = ConversationHandler(
        entry_points=[CommandHandler("clone", clone_start), CommandHandler("clonar", clone_start)],
        states={
            CLONE_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, clone_receive_name)],
            CLONE_LANGUAGE: [
                CallbackQueryHandler(clone_pick_language, pattern=r"^langpick:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, clone_expect_language_button),
            ],
            CLONE_DESCRIPTION: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, clone_receive_description),
                CommandHandler("skip", clone_skip_description),
            ],
            CLONE_TAGS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, clone_receive_tags),
                CommandHandler("skip", clone_skip_tags),
            ],
            CLONE_NOISE: [
                CallbackQueryHandler(clone_pick_noise, pattern=r"^clonenoise:"),
                MessageHandler(filters.TEXT & ~filters.COMMAND, clone_expect_noise_button),
            ],
            CLONE_SAMPLES: [
                MessageHandler(
                    filters.AUDIO | filters.VOICE | filters.Document.ALL,
                    clone_receive_sample,
                ),
                MessageHandler(filters.TEXT & ~filters.COMMAND, clone_expect_samples),
                CommandHandler("clone_done", clone_finish),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
        per_chat=True,
        per_user=True,
        allow_reentry=True,
    )

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("settings", settings_command))
    app.add_handler(CommandHandler("apikey", apikey_command))
    app.add_handler(CommandHandler("model", model_command))
    app.add_handler(CommandHandler("speed", speed_command))
    app.add_handler(CommandHandler("temp", temperature_command))
    app.add_handler(CommandHandler("temperature", temperature_command))
    app.add_handler(CommandHandler("encoding", encoding_command))
    app.add_handler(CommandHandler("samplerate", samplerate_command))
    app.add_handler(CommandHandler("normalize", normalize_command))
    app.add_handler(CommandHandler("timestamps", timestamps_command))
    app.add_handler(CommandHandler("voices", voices_command))
    app.add_handler(CommandHandler("voice", voice_command))
    app.add_handler(CommandHandler("myvoices", myvoices_command))
    app.add_handler(clone_handler)
    app.add_handler(CommandHandler("cancel", cancel_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(filters.Document.ALL & ~filters.COMMAND, document_message_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
    return app


def main() -> None:
    app = build_application()
    logger.info("Bot oficial da Inworld iniciado.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
