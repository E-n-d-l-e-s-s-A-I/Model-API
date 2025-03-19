import copy
from fastapi import FastAPI
from model import (
    Model,
    CONVERSATION_F0R_ENG_TO_RU_TRANSLATE,
    CONVERSATION_F0R_RU_TO_ENG_TRANSLATE,
    CONVERSATION_F0R_EXTRACT
)
import uvicorn


app = FastAPI(
    title="Model API",
    description="API для перевода текста на русский и английский языки",
    version="1.0.0",
)


@app.get("/extract")
async def extract(text: str) -> list[str]:
    """Переводит текст на русский язык по контексту диалога."""

    conversation = copy.copy(CONVERSATION_F0R_EXTRACT)
    conversation[-1] += text

    return Model.generate(conversation, max_new_tokens=len(text)).split("\n")


@app.get("/translate_ru_to_eng")
async def translate_ru_to_eng(text: str) -> str:
    """Переводит текст на английский язык."""

    conversation = copy.copy(CONVERSATION_F0R_RU_TO_ENG_TRANSLATE)
    conversation[-1] += text

    return Model.generate(conversation, max_new_tokens=len(text))


@app.get("/translate_eng_to_ru")
async def translate_eng_to_ru(text: str) -> str:
    """Переводит текст на русский язык."""

    conversation = copy.copy(CONVERSATION_F0R_ENG_TO_RU_TRANSLATE)
    conversation[-1] += text

    return Model.generate(conversation, max_new_tokens=len(text))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
