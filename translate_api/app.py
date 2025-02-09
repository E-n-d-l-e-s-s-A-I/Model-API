import copy
from fastapi import FastAPI
from model import (
    Model,
    CONVERSATION_F0R_ENG_TO_RU_TRANSLATE,
    CONVERSATION_F0R_RU_TO_ENG_TRANSLATE,
)
from schemas import Conversation
import uvicorn


app = FastAPI(
    title="Translate API",
    description="API для перевода текста на русский и английский языки",
    version="1.0.0",
)


@app.post("/translate_by_conversation")
async def translate_by_conversation(conversation: Conversation) -> str:
    """Переводит текст на русский язык по контексту диалога."""

    conversation = [
        ": ".join([message.role, message.content])
        for message in conversation.conversation
    ]
    return Model.generate(conversation)


@app.get("/translate_ru_to_eng")
async def translate_ru_to_eng(text: str) -> str:
    """Переводит текст на английский язык."""

    conversation = copy.copy(CONVERSATION_F0R_RU_TO_ENG_TRANSLATE)
    conversation[-1] += text

    return Model.generate(conversation)


@app.get("/translate_eng_to_ru")
async def translate_eng_to_ru(text: str) -> str:
    """Переводит текст на русский язык."""

    conversation = copy.copy(CONVERSATION_F0R_ENG_TO_RU_TRANSLATE)
    conversation[-1] += text

    return Model.generate(conversation)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
