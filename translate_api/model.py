import transformers
import torch
from huggingface_hub import login
from settings import settings
import os

# для ускорения работы модели
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MODEL_PATH = settings.MODEL_PATH

if settings.HF_TOKEN:
    login(settings.HF_TOKEN)


CONVERSATION_F0R_RU_TO_ENG_TRANSLATE = [
    "system: Ты переводчик текстов с русского языка на английский.",
    "user: Переведи данные фрагменты текста с русского языка на английский.\nФрагменты:\nИскусственный интеллект — универсальный инструмент.\nЕстественный интеллект",
    "assistant: Artificial intelligence is a universal tool.\nNatural intelligence",
    "user: Переведи данные фрагменты текста с русского языка на английский.\nФрагменты:\n",
]

CONVERSATION_F0R_ENG_TO_RU_TRANSLATE = [
    "system: Ты переводчик текстов с ангийского языка на русский.",
    "user: Переведи данные фрагменты текста с английского языка на русский.\nФрагменты:\nArtificial intelligence is a universal tool.\nNatural intelligence",
    "assistant: Искусственный интеллект — универсальный инструмент.\nЕстественный интеллект",
    "user: Переведи данные фрагменты текста с английского языка на русский.\nФрагменты:\n",
]


class Model:
    """Модель для перевода текста на русский и английский языки."""

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    @classmethod
    def generate(cls, conversation: list[str]):
        """Генерирует ответ на основе контекста диалога."""

        prompt = "\n".join(conversation) + "\nassistant:"

        response = Model.pipeline(
            prompt,
            max_new_tokens=50,
            do_sample=False,
        )
        assistant_reply: str = response[0]["generated_text"][len(prompt) :].strip()
        return assistant_reply.split("\nuser")[0]
