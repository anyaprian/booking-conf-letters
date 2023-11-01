# booking-conf-letters

# faqtrip-content-processing

## Сервис решает задачу автоматического парсинга информации из электронных писем

Сервис работает с простым текстом, html и pdf.

---

### API Usage

#### 1. Парсинг информации

- **Endpoint:** https://content-parser.k8s.faqtrip.com/parse_content
- **Method:** POST
- **Request Format:** Словарь с по крайней мере одним из элементов письма: «text/plain», «text/html», «application/pdf».
  
  [Пример запроса](https://www.notion.so/faqtrip/db51ca3f682c424cb1e1e4f95eabbd68?pvs=4#9ca8bd8b7fb341cf9d6c6ecc6364f3ce)

- **Response Format:** Список найденных бронирований (текст бронирования и ключевая информация).

  [Пример ответа](https://www.notion.so/faqtrip/db51ca3f682c424cb1e1e4f95eabbd68?pvs=4#0cdcb6970b53493ab4f42d17b7448884)

#### 2. Health Check

- **Endpoint:** https://content-parser-dev.k8s.faqtrip.com/api/v1/healthcheck
- **Method:** GET

---

### Локальный старт с Docker

#### 1. Клонирование репозитория

```bash
git clone https://github.com/faqtrip/faqtrip-content-processing.git
cd faqtrip-content-processing
```

#### 2. Сборка образа и запуск контейнера

```bash
docker compose up 
```

После запуска, сервис будет доступен по адресу: http://localhost:8080
