# HypoSpace Roadmap (MVP 6–8 недель)

## Цель MVP
Собрать лёгкий end-to-end пайплайн, который превращает активации PyTorch-модели в интерпретируемые и проверяемые концепты:

`model -> activations -> hierarchical kernels -> semantic labels -> mechanistic checks -> governance scorecard -> UI`

---

## Этап 1 (Недели 1–2): Foundation + E2E Skeleton

### Deliverables
- Базовая структура проекта:
  - `core/`, `data/`, `interpretability/`, `viz/`
  - `api.py`, `main.py`, `tests/`
- `data/extractor.py`:
  - nnsight hooks для извлечения активаций
  - кэширование (diskcache/joblib)
- `core/decoder.py` (v0): единая точка входа `decode(...)`
- CLI smoke-run, который сохраняет артефакты в локальное хранилище

### Acceptance Criteria
- На 10–50 примерах система извлекает активации и создаёт первичный набор фич.
- Команда может запустить демо без ручных патчей к коду.

---

## Этап 2 (Недели 3–4): Hierarchical Kernels + Persistence

### Deliverables
- `core/hierarchy.py`:
  - Matryoshka SAE wrapper (приоритетный backend)
  - fallback-режим для компактных словарей на CPU
- `core/kernel_library.py`:
  - `KernelTemplate`
  - versioning (semver + метаданные тренировки)
  - save/load/match/merge API
- Первичное cross-run сопоставление концептов

### Acceptance Criteria
- Повторный запуск на той же модели восстанавливает и сопоставляет ключевые концепты.
- CPU fallback работает в ограниченном режиме (малые словари, batch=1).

---

## Этап 3 (Недели 5–6): Semantic + Mechanistic + Governance

### Deliverables
- `interpretability/semantic.py`:
  - template-based auto-interpretation
  - optional Narrator (LiteLLM/local LLM)
- `interpretability/mechanistic.py`:
  - pyvene interventions
  - activation patching для top-k features
  - baseline linear probing
- `interpretability/faithfulness.py`:
  - intervention-based checks
  - governance scorecard (faithfulness/stability/risk flags)

### Acceptance Criteria
- Для выбранных фич доступны: описание, интервенционный результат, scorecard.
- Низкая достоверность явно маркируется в отчёте.

---

## Этап 4 (Недели 7–8): Visualization + Hardening

### Deliverables
- `viz/streamlit_app.py` с вкладками:
  - Kernel Explorer
  - Semantic Canvas
  - Mechanistic Probes + Governance
- `viz/canvas.py`:
  - визуализация иерархии/связей/силы активации
- Тесты:
  - smoke E2E
  - контрактные тесты форматов данных
  - регрессионные тесты на фиксированном мини-наборе
- Quickstart документация

### Acceptance Criteria
- Новый пользователь поднимает demo и получает рабочий отчёт без глубокой настройки.
- Ключевые сценарии проходят smoke + regression проверки.

---

## KPI MVP
- **Time-to-first-insight**: первый валидный отчёт < 15 минут на малом наборе.
- **Concept consistency**: стабильное сопоставление концептов между повторными запусками.
- **Faithfulness coverage**: >= 80% top features имеют mechanistic check.
- **CPU viability**: pipeline запускается без GPU в ограниченном профиле.

---

## Backlog после MVP
- Полноценный USAE alignment для межмодельного универсального пространства концептов.
- Расширенные causal path tracing сценарии.
- Библиотека переиспользуемых Persistent Kernels между семействами моделей.
- Экспорт governance scorecard в стандартизированные отчёты.
