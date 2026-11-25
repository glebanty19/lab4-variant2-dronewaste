# Лабораторна робота №4 — варіант 2

Код для виявлення куп будівельних відходів на аерознімках DroneWaste за допомогою Faster R-CNN ResNet-50 FPN. У репозиторії є скрипти підготовки даних, навчання, оцінювання, аналізу помилок і створення звіту. Датасет, ваги моделей, метрики та DOCX не зберігаються в Git.

## Встановлення й дані

```powershell
py -m pip install -r requirements.txt
py lab4/download_dronewaste_parallel.py
```

Джерело: [DroneWaste v1.0](https://zenodo.org/records/17045559). Завантажувач перевіряє контрольні суми й створює `datasets/dronewaste/images/` та `datasets/dronewaste/dronewaste_v1.0.json` у корені репозиторію.

## Запуск

```powershell
py lab4/run_dronewaste_lr4.py
```

Скрипт готує COCO-підмножини, навчає контрольну й основну моделі, вибирає checkpoint за validation, оцінює test і вплив масштабу та плиток, генерує локальний DOCX. Результати й чекпойнти з’являться в `lab4/runs/`, підготовлені дані — у `lab4/data/`.

Для ЛР6 потрібні `lab4/runs/variant2_audit/selected_lr4_model.json` і вибраний у ньому `best.pt`. Код ЛР6 зберігається окремо: [lab6-variant2-trudi](https://github.com/glebanty19/lab6-variant2-trudi).

Вихідні фрагменти мають 640×640 px; фізичний GSD і повні ортомозаїки у використаному архіві відсутні. Ці межі зазначено у звіті.
