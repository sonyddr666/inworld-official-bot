# Inworld Official Telegram Bot

Bot Telegram separado do projeto `webscrap-tts`, mas usando a API oficial da Inworld com `Authorization: Basic ...`.

## O que ele faz

- Salva API key oficial por usuario com `/apikey`
- Lista vozes da Inworld com filtro por idioma
- Alterna entre vozes de sistema e vozes clonadas
- Clona voz via Telegram com ate 3 amostras
- Apaga vozes clonadas pelo menu `/myvoices`
- Recebe texto curto, texto longo e arquivos `.txt` ou `.md`
- Divide textos grandes em partes automaticamente e avisa sobre custo/latencia
- Envia audio reproduzivel e baixavel, em MP3 por padrao
- Permite trocar modelo, velocidade, temperatura, encoding, sample rate, normalizacao e timestamps

## Instalar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Preencha no `.env`:

```env
TELEGRAM_BOT_TOKEN=
INWORLD_BASE64_CREDENTIAL=
INWORLD_DEFAULT_VOICE_ID=
INWORLD_MAX_CHARS_PER_REQUEST=1900
LOG_LEVEL=INFO
```

## API key

O bot aceita:

- `/apikey Basic BASE64_AQUI`
- `/apikey key_id:key_secret`
- `/apikey clear`

Se quiser uma chave padrao para todos os usuarios do bot, preencha `INWORLD_BASE64_CREDENTIAL` no `.env`.

## Rodar

```bash
python bot.py
```

## Comandos

- `/start`
- `/help`
- `/apikey`
- `/settings`
- `/voices`
- `/voice`
- `/myvoices`
- `/clone`
- `/model`
- `/speed`
- `/temp`
- `/encoding`
- `/samplerate`
- `/normalize`
- `/timestamps`
- `/cancel`

## Clone de voz

Fluxo:

1. `/clone`
2. Informar nome
3. Escolher idioma
4. Opcionalmente informar descricao e tags
5. Escolher se remove ruido
6. Enviar de 1 a 3 audios
7. Finalizar com `/clone_done`

Se o audio tiver `caption`, o texto sera enviado como `transcription`.

## Textos grandes

- O bot divide automaticamente em blocos de ate `INWORLD_MAX_CHARS_PER_REQUEST`
- Se o texto for muito grande, ele avisa antes de comecar
- Para textos maiores que o limite de mensagem do Telegram, envie um `.txt`

## Observacoes

- O armazenamento local das configuracoes fica em `data/user_state.json`
- O bot usa MP3 por padrao porque toca melhor dentro do Telegram
- Encodings nao-playable sao enviados como documento
