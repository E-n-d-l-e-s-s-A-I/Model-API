# Model-API

## Запуск
- работает на ubuntu. В контейнере все падало. На винде все падало. ML боль. 
- Создать виртуальное окружение и установить зависимости(рекомендуемая версия python на которой происходила разработка 3.12)
- Задать переменные окружения HF_TOKEN, MODEL_PATH. MODEL_PATH может быть путем к локальной модели.
- `cd model_api`
- `python app.py`
- api будет доступен - `http://localhost:8002`


## Пример запроса

```bash
curl -X 'POST' \
  'http://localhost:8002/synonyms' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "term": "головная боль"
}'
```
	
Response body:

```json
[
  {
    "term": "Боль в голове"
  },
  {
    "term": "Кластерная головная боль"
  },
  {
    "term": "Тензионная головная боль"
  },
  {
    "term": "Посттравматический головной боль."
  },
  {
    "term": "Синусная головная боль"
  },
  {
    "term": "Гипнический головокружение"
  },
  {
    "term": "Заболевания головной боли"
  },
  {
    "term": "Головная боль"
  },
  {
    "term": "Головная боль, вызванная поражением шейного отдела позвоночника."
  },
  {
    "term": "Временная головная боль"
  },
  {
    "term": "Боль в голове"
  },
  {
    "term": "Первичный головная боль от кашля."
  },
  {
    "term": "Передний лобный болевой синдром"
  },
  {
    "term": "Мышечная головная боль"
  },
  {
    "term": "Первичный головная боль от пронзения."
  },
  {
    "term": "Первичный Тромбоз Головы"
  },
  {
    "term": "Пост-спинномозговая головная боль"
  },
  {
    "term": "Боль в голове повторяющаяся"
  },
  {
    "term": "Хроническая головная боль"
  },
  {
    "term": "Перепады головной боли"
  },
  {
    "term": "Частые головные боли"
  },
  {
    "term": "Отскокная головная боль с непреодолимой головной болью"
  },
  {
    "term": "Отскок головной боли без непреодолимой головной боли"
  },
  {
    "term": "Окулярная головная боль"
  },
  {
    "term": "Головная боль полнота"
  }
]
```
