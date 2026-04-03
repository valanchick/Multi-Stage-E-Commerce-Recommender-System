# Multi-Stage E-Commerce Recommender System (Deep Learning)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=yandex&logoColor=black)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![System Design](https://img.shields.io/badge/Architecture-Two--Tower_DSSM-blue)

## Бизнес-цель
Проектирование и реализация полномасштабного 3-уровневого конвейера рекомендаций (Retrieval $\rightarrow$ Ranking $\rightarrow$ Re-ranking) для крупного интернет-магазина. 

Цель системы — за миллисекунды отбирать релевантные товары из огромного каталога, максимизируя вероятность покупки (Conversion Rate), при этом сохраняя категориальное разнообразие (Diversity) в итоговой выдаче, чтобы избежать эффекта "каннибализации" витрины. Обучение проводилось на реальном датасете **REES46 eCommerce behavior** (1+ млн взаимодействий).

## Ключевые архитектурные и инженерные решения

В проекте решены классические проблемы высоконагруженных (High-Load) рекомендательных систем:

* **Strict Time-based Split:** В рекомендациях случайное разбиение (Random Split) ведет к утечке данных (Data Leakage). Данные были строго отсортированы по `event_time`, после чего последние 20% времени были выделены в тестовую выборку. Это симулирует реальное поведение модели в продакшене.
* **On-the-fly Negative Sampling:** Для обучения Retrieval-модели был написан кастомный `torch.utils.data.Dataset`. Он динамически генерирует негативные примеры (Negative Samples) во время формирования батчей, что позволило сэкономить гигабайты оперативной памяти.
* **Rich Two-Tower Architecture (Late Fusion):** Спроектирована архитектура "Двух башен" на PyTorch. Башня товара конкатенирует эмбеддинги `product_id`, `brand` и `category_code`. Это решает проблему холодного старта (Cold Start): модель способна рекомендовать совершенно новые товары исключительно на основе их метаданных.
* **Business-Weighted Loss:** Использована функция потерь `BCEWithLogitsLoss(reduction='none')`. Ошибка сети динамически взвешивалась в зависимости от ценности микроконверсии пользователя (view = 1, cart = 2, purchase = 3), заставляя градиенты сильнее реагировать на реальные покупки.

## Архитектура пайплайна (Multi-Stage)

### Stage 1: Candidate Generation (Retrieval)
* **Модель:** PyTorch Two-Tower Neural Network (DSSM).
* **Логика:** Модель вычисляет косинусное расстояние (Dot Product после `F.normalize`) между латентным вектором пользователя и векторами товаров. На инференсе модель работает как эмулятор векторной БД (FAISS), отбирая Топ-100 релевантных кандидатов из всего каталога за миллисекунды.

### Stage 2: Ranking
* **Модель:** CatBoost Ranker.
* **Логика:** Топ-100 кандидатов обогащаются "тяжелыми" признаками (история пользователя, цена, скор нейросети). Обучение проводилось с использованием функции потерь **YetiRank**. В отличие от LogLoss, YetiRank оптимизирует метрики ранжирования (NDCG), фокусируясь на правильном порядке сортировки Топ-50 товаров.

### Stage 3: Re-ranking (Бизнес-правила)
* **Алгоритм:** MMR (Maximal Marginal Relevance).
* **Логика:** Балансировка итогового Топ-10. Алгоритм штрафует товары-кандидаты, если их категория уже присутствует в выдаче. Это гарантирует, что пользователь увидит разнообразную витрину (например, смартфон, наушники и чехол), а не 10 одинаковых смартфонов разных цветов.

## Как запустить

Данный репозиторий автоматически скачивает необходимые данные с Kaggle через `kagglehub`. Вам не нужно загружать гигабайты файлов вручную.

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/valanchick/Multi-Stage-E-Commerce-Recommender-System.git
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Запустите Jupyter Notebook:
   ```bash
   jupyter notebook notebooks/recsys_pipeline.ipynb
   ```
