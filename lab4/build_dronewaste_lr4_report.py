#!/usr/bin/env python3
"""Build the subject-matched LR4 variant 2 report from DroneWaste results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from report_utils import add_code_link, add_table, num
from sample_report_style import style_report

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/dronewaste_report_artifacts"
FINE="dronewaste_piles_fine"
BASE="dronewaste_piles_head"
CLASSES={"Rubble":"Будівельний щебінь / завали", "Construction and demolition materials":"Будівельні та демонтажні матеріали"}


def load(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))


def metric(run,mode):
    return load(f"runs/{run}/{mode}/{mode}_metrics.json")


def figure(doc,path,caption,width=6.2):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ROOT/path),width=Inches(width))
    p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.italic=True;r.font.size=Pt(9)


def make_charts(fine,base,x2,x4,tiles,sizes):
    OUT.mkdir(parents=True,exist_ok=True)
    histories=[(load(f"runs/{BASE}/history.json"),"Head only"),
               (load(f"runs/{FINE}/history.json"),"Fine tuning")]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for rows,name in histories:
        axes[0].plot([r["epoch"] for r in rows],[r["train_loss"] for r in rows],marker="o",label=name)
        axes[1].plot([r["epoch"] for r in rows],[r["val_map50_95"] for r in rows],marker="o",label=name)
    axes[0].set(xlabel="Epoch",ylabel="Train loss");axes[1].set(xlabel="Epoch",ylabel="Validation mAP@[.5:.95]")
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/"training.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    factors=[1,2,4]
    axes[0].plot(factors,[m["map50_95"] for m in (fine,x2,x4)],marker="o",label="Whole fragment")
    axes[0].scatter([1],[tiles["map50_95"]],marker="s",label="Tile 320 at ×1")
    axes[0].set(xlabel="Simulated pixel-size multiplier",ylabel="mAP@[.5:.95]",title="Resolution and tiles")
    axes[0].set_xticks(factors)
    for size in ("small","medium","large"):
        axes[1].plot(factors,[sizes["area_ap_by_scale"][key][size] for key in ("x1","x2","x4")],marker="o",label=size)
    axes[1].set(xlabel="Simulated pixel-size multiplier",ylabel="AP@[.5:.95]",title="Object area")
    axes[1].set_xticks(factors)
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/"resolution.png",dpi=160);plt.close(fig)


def main():
    global FINE
    selection_path=ROOT/"runs/variant2_audit/selected_lr4_model.json"
    selection=json.loads(selection_path.read_text(encoding="utf8")) if selection_path.is_file() else None
    if selection:FINE=selection["run"]
    audit=load("runs/variant2_audit/dataset_audit.json")["lr4_exact"]
    prep=load("data/dronewaste_piles/preparation.json")
    sizes=load("runs/variant2_audit/lr4_exact_size_errors.json")
    examples=load("runs/variant2_audit/lr4_exact_error_examples.json")
    fine=metric(FINE,"test");base=metric(BASE,"test")
    fine_val=metric(FINE,"val");base_val=metric(BASE,"val")
    x2=metric(FINE,"test_x2");x4=metric(FINE,"test_x4");tiles=metric(FINE,"test_tiles")
    make_charts(fine,base,x2,x4,tiles,sizes)
    doc=Document();sec=doc.sections[0]
    sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7)
    sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
    doc.styles["Normal"].font.name="Arial";doc.styles["Normal"].font.size=Pt(10)
    for name in ("Title","Heading 1","Heading 2"):
        doc.styles[name].font.name="Arial";doc.styles[name].font.color.rgb=RGBColor(31,78,121)
    sec.header.paragraphs[0].text="Лабораторна робота № 4 · Детекція на аерознімках"
    sec.footer.paragraphs[0].text="Варіант 2 · DroneWaste"
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(90)
    r=p.add_run("ЛАБОРАТОРНА РОБОТА № 4");r.bold=True;r.font.size=Pt(22)
    p=doc.add_paragraph("Виявлення куп будівельних відходів на аерознімках\nВаріант 2")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.font.size=Pt(16)
    p=doc.add_paragraph("Дисципліна: Штучний інтелект для циркулярної економіки\nВиконав: Бондар Глєб Ігорович, КНм-26-1\nХмельницький, 2026")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    add_code_link(doc)
    doc.add_heading("1. Мета та відповідність завданню",level=1)
    doc.add_paragraph("Варіант 2 вимагає детекції куп будівельних відходів на аерознімках Faster R-CNN ResNet-50 FPN, дослідження зменшення роздільності ×2 і плиток. Набір DroneWaste містить зображення з ортомозаїк БПЛА та COCO рамки. Як цільові підкласи обрано Rubble (щебінь/завали) і Construction and demolition materials (будівельні та демонтажні матеріали). Інші 18 класів не є цільовими й виступають фоном.")
    doc.add_heading("2. Підготовка даних",level=1)
    doc.add_paragraph("Вихідний DroneWaste (CC BY 4.0) містить 4993 зображення й 5135 анотацій двадцяти класів із 17 локацій. Для цього досліду site5 повністю відведено у validation, site17 — у test. Train складається з усіх кадрів інших локацій, де є цільові об’єкти, та випадково відібраних негативних кадрів у співвідношенні 2:1 до позитивних кадрів (seed 42). Так зменшено перевагу порожніх кадрів без витоку з test. Вихідні COCO bbox збережено без зміни координат, а файли зображень зв’язано жорсткими посиланнями з вихідним архівом.")
    add_table(doc,["Підмножина","Зображень","Rubble","C&D materials","Малі","Середні","Великі"],
              [[split,str(audit["splits"][split]["images"]),
                str(audit["splits"][split]["classes"].get("Rubble",0)),
                str(audit["splits"][split]["classes"].get("Construction and demolition materials",0)),
                *[str(audit["splits"][split]["sizes_coco"].get(size,0)) for size in ("small","medium","large")]]
               for split in ("train","val","test")])
    add_table(doc,["Підмножина","Ширина bbox, px (min/med/max)","Висота bbox, px (min/med/max)","Частка малих/середніх/великих"],
              [[split,
                "/".join(num(audit["splits"][split]["box_width_px"][k]) for k in ("min","median","max")),
                "/".join(num(audit["splits"][split]["box_height_px"][k]) for k in ("min","median","max")),
                "/".join(f"{100*audit['splits'][split]['sizes_coco'].get(k,0)/max(1,audit['splits'][split]['annotations']):.1f}%" for k in ("small","medium","large"))]
               for split in ("train","val","test")])
    bins=("<32","[32,64)","[64,128)","[128,256)","[256,512)","≥512")
    add_table(doc,["Підмножина","Сторона, px",*bins],
              [[split,side,*[str(audit["splits"][split][f"box_{key}_hist_px"][bucket]) for bucket in bins]]
               for split in ("train","val","test") for side,key in (("Ширина","width"),("Висота","height"))])
    doc.add_paragraph("Малі об’єкти: bbox площа <32² px; середні: 32²–96² px; великі: ≥96² px. Перевірено існування всіх використаних файлів, межі й додатність bbox, правильність категорій та відсутність повтору назв файлів між підмножинами. Поділ виконано за локаціями, а не випадково за сусідніми фрагментами однієї ортомозаїки.")
    figure(doc,"runs/variant2_audit/dronewaste_18_boxes.png","Рисунок 1 — 18 зображень із COCO рамками: шість на кожну підмножину.",6.5)
    doc.add_heading("3. Навчання та протокол оцінювання",level=1)
    chosen_training={"dronewaste_piles_fine":"Повне донавчання від COCO; SGD lr=0,005; стандартне перемішування; patience=2",
                     "dronewaste_piles_weighted_fine":"Повне донавчання від COCO; SGD lr=0,005; позитивні кадри з вагою 4; patience=4",
                    }.get(FINE,FINE)
    add_table(doc,["Параметр","Значення"],[["Детектор","Faster R-CNN ResNet-50 FPN; COCO pretrained"],
              ["Вхід","640×640 px; RGB; batch size 1"],["Класи","Фон + Rubble + Construction and demolition materials"],
              ["Аугментація","Горизонтальне віддзеркалення p=0,5"],["Оптимізація","SGD momentum=0,9, weight decay=0,0005; seed 42"],
              ["Baseline","Заморожено backbone/RPN; навчено нову класифікаційно-регресійну голову"],
              ["Основна модель",chosen_training],
              ["Вибір","Найкращий checkpoint за validation mAP@[0.5:0.95]; зупинка за validation"],
              ["Поріг","Максимум micro-F1 на validation; test не використано для добору"]])
    p=doc.add_paragraph("Код: category_id → {1:Rubble, 2:C&D}; box=[x,y,x+w,y+h]; loss=sum(model(images,targets).values()); best.pt=argmax(val_mAP@[.5:.95]).")
    for r in p.runs:r.font.name="Consolas";r.font.size=Pt(9)
    histories={run:load(f"runs/{run}/history.json") for run in (BASE,FINE)}
    add_table(doc,["Модель","Епох","Краща епоха","Val mAP@0,5","Val mAP@[.5:.95]","Test mAP@[.5:.95]"],
              [[name,str(len(histories[run])),str(max(histories[run],key=lambda r:r["val_map50_95"])["epoch"]),
                num(val["map50"]),num(val["map50_95"]),num(test["map50_95"])]
               for name,run,val,test in (("Голова",BASE,base_val,base),("Повне донавчання",FINE,fine_val,fine))])
    if base["map50_95"]>fine["map50_95"]:
        doc.add_paragraph("На відкладеній test локації контрольна модель із замороженим екстрактором дала трохи вищий mAP, попри нижчий validation mAP. Це ознака нестабільного переносу між локаціями; checkpoint не переобрано за test, щоб не допустити витоку результату в протокол відбору.")
    if selection:
        add_table(doc,["Кандидат","Протокол","Validation mAP@[.5:.95]"],
                  [[row["run"],row["description"],num(row["val_map50_95"])] for row in selection["candidates"]])
        doc.add_paragraph("Основний checkpoint обрано за validation mAP@[.5:.95]. Додаткові спроби навчання наведено для прозорості; test не використовували для вибору. Baseline із навчанням лише голови має власний протокол і служить точкою відліку.")
    figure(doc,"runs/dronewaste_report_artifacts/training.png","Рисунок 2 — Train loss і validation mAP за епохами.")
    doc.add_heading("4. Результати test і розміри",level=1)
    add_table(doc,["Клас","GT","AP@0,5","AP@[.5:.95]"],
              [[CLASSES[name],str(fine["classes"][name]["support"]),num(fine["classes"][name]["ap50"]),num(fine["classes"][name]["ap50_95"])]
               for name in CLASSES]+[["mAP","—",num(fine["map50"]),num(fine["map50_95"])]])
    op=fine["operating_point"]
    doc.add_paragraph(f"Поріг {num(fine_val['best_f1_threshold']['confidence'])} обрано за validation micro-F1. На test precision={num(op['precision'])}, recall={num(op['recall'])}, F1={num(op['f1'])}. Пропущена купа може лишитися поза екологічним обліком, а хибну тривогу можна перевірити вручну; тому в практичному сценарії пропуск дорожчий. Водночас наведений поріг оптимізує саме F1, а не грошову функцію вартості, і за низьких precision/recall автоматичне впровадження не обґрунтоване. AP обчислено власною 101-точковою COCO-подібною реалізацією; вона не є повною заміною pycocotools.")
    figure(doc,f"runs/{FINE}/test/test_pr.png","Рисунок 3 — PR-криві на test.")
    add_table(doc,["Розмір","GT на test","AP@[.5:.95]"],
              [[size,str(sum(v["support"] for v in sizes["area_ap"][size].values() if isinstance(v,dict))),
                num(sizes["area_ap"][size]["mAP50_95"])] for size in ("small","medium","large")])
    small_support=sum(v["support"] for v in sizes["area_ap"]["small"].values() if isinstance(v,dict))
    if small_support<10:
        doc.add_paragraph(f"У test лише {small_support} малих цільових рамок; AP цієї групи має дуже високу невизначеність і не придатний для сильних висновків про дрібні купи.")
    doc.add_heading("5. Роздільність і плитки",level=1)
    add_table(doc,["Режим","mAP@0,5","mAP@[.5:.95]","AP малих","AP середніх","AP великих"],
              [[title,num(row["map50"]),num(row["map50_95"]),
                *[num(sizes["area_ap_by_scale"].get(key,{}).get(size)) for size in ("small","medium","large")]]
               for title,key,row in (("Вихідний", "x1", fine),("Зменшення ×2","x2",x2),("Зменшення ×4","x4",x4),
                                     ("Плитка 320 px, 20% overlap","tiles",tiles))])
    figure(doc,"runs/dronewaste_report_artifacts/resolution.png","Рисунок 4 — AP від відносного збільшення пікселя та розміру об’єкта.")
    relative_drop=1-x2["map50_95"]/fine["map50_95"] if fine["map50_95"] else None
    if relative_drop is not None:
        doc.add_paragraph(f"За переходу від ×1 до ×2 відносне падіння mAP@[.5:.95] становило {100*relative_drop:.1f}%. Якщо припустиме падіння якості становить 20%, ×2 для цієї моделі не проходить критерій. Правило з ЛР1 про співвідношення розміру об’єкта й GSD можна зіставити лише відносно: після ×2 купа займає приблизно вдвічі менше пікселів за кожною стороною. Абсолютної межі GSD у сантиметрах набір не задає.")
    doc.add_paragraph("Зменшення виконано BOX downsample із поверненням до 640×640 bilinear; GT рамки не змінюються. Плитки 320 px із 20% перекриттям оброблено окремо й об’єднано NMS при IoU 0,5. Вихідні фрагменти DroneWaste мають 640×640 px, а повні ортомозаїки не входять у відкритий архів; отже, це порівняння на фрагментах, не на повній великій сцені. Фізичний GSD кадрів у COCO метаданих відсутній, тому мінімальний допустимий GSD у сантиметрах обґрунтувати неможливо; можна визначити лише мінімальну деталізацію за вибраним порогом AP/recall.")
    doc.add_heading("6. Помилки та приклади",level=1)
    add_table(doc,["Категорія","Кількість"],[[kind,str(sizes["errors"].get(kind,0))]
              for kind in ("class","localization","class_and_localization","duplicate","background","miss")])
    add_table(doc,["Розмір GT","Пропуски","Частка GT"],
              [[size,str(sizes["misses_by_size"].get(size,0)),
                num(sizes["misses_by_size"].get(size,0)/max(1,sum(v["support"] for v in sizes["area_ap"][size].values() if isinstance(v,dict))))]
               for size in ("small","medium","large")])
    figure(doc,"runs/variant2_audit/lr4_exact_size_errors.png","Рисунок 5 — AP за розмірами та категорії помилок.")
    for kind in ("tp","class","localization","class_and_localization","duplicate","background","miss"):
        if examples.get(kind):
            figure(doc,f"runs/variant2_audit/lr4_exact_examples_{kind}.png",
                   f"Приклади {kind}: {len(examples[kind])} кадрів, червона рамка — прогноз, зелена — GT.",6.5)
    doc.add_paragraph("Помилки класу й локалізації визначено порогами IoU 0,1 і 0,5; дублікати — прогнози до вже зіставленого GT; фон — IoU<0,1. Для кожного типу показано до десяти різних кадрів; якщо подій менше десяти, усі доступні. Імовірні причини: малий масштаб, неоднорідні межі куп, сусідні матеріали, тіні й неповна розмітка деяких фрагментів. Рекомендації: більше незалежних локацій, точніші рамки, масштабна аугментація, більший вхід і контекст на межах плиток.")
    doc.add_heading("7. Паспорт експерименту",level=1)
    add_table(doc,["Параметр","Фактичне значення"],[["Джерело","DroneWaste v1.0, Zenodo 17045559"],
             ["Поділ","Site5 validation, site17 test, решта site train; негативи 2:1"],
             ["Дані","Два класи куп, COCO bbox, 640×640 RGB"],
             ["Модель","Faster R-CNN ResNet-50 FPN, COCO weights"],
             ["Навчання",f"seed42; batch1; horizontal flip p0,5; {chosen_training}"],
             ["Оцінювання","Validation checkpoint/threshold; test AP@.5, mAP@[.5:.95], AP size"],
             ["Ціна помилок","FN: не облікована купа; FP: ручна перевірка; робочий поріг — максимум validation F1"],
             ["Відтворення","prepare_dronewaste_lr4.py, cdw_detector.py, analyze_lr4.py"],
             ["Обмеження","Відкрито лише 640 px фрагменти; фізичний GSD та повні ортомозаїки відсутні"]])
    doc.add_heading("8. Використання ШІ",level=1)
    doc.add_paragraph("ChatGPT/Codex використано для зіставлення вимог з даними, підготовки скрипту поділу, перевірки формату COCO, аналізу метрик і структури звіту. Приклади запитів: «Які класи DroneWaste відповідають купам будівельних відходів?», «Як відокремити локації між train/validation/test?», «Як обчислити AP за розмірами COCO?». Рекомендації: залишити Rubble і Construction and demolition materials, розділяти за site, добирати поріг лише на validation. Перевірка: локальний аудит рамок і 18 зображень, порівняння JSON із таблицями та повторний запуск test. Власний висновок: кадри 640 px дозволяють оцінити лише плитки всередині фрагмента; фізичний GSD і повні сцени недоступні. Авторові слід самостійно переглянути код і приклади перед захистом.")
    doc.add_heading("9. Відповіді на контрольні питання",level=1)
    qa=[("1. Детектування і класифікація","Детектування повертає координати і класи багатьох об’єктів, класифікація — мітку кадру. Рамки задають кутами або центром, орієнтована рамка містить кут."),
        ("2. IoU і NMS","IoU — перетин/об’єднання площ. NMS прибирає слабші дубльовані рамки з великим IoU."),
        ("3. Faster R-CNN","ResNet-FPN формує ознаки, RPN пропонує якірні рамки, RoI Align виділяє ознаки, голова прогнозує клас і поправки центру та логарифмів розміру."),
        ("4. Одно- і двоетапні моделі","Двоетапні спершу пропонують регіони, одноетапні одразу прогнозують рамки. Focal loss пригнічує внесок легких негативів."),
        ("5. Anchor-based і anchor-free","Faster R-CNN використовує anchors. FCOS і YOLOv8 прогнозують відстані/рамки від точок карти ознак без заздалегідь заданих якорів."),
        ("6. Об’єкти та труднощі аерознімків","Купи, контейнери, техніка та будівлі мають різні масштаби; тіні, щільність, фон і межі плиток ускладнюють виявлення."),
        ("7. GSD і розмір входу","Піксельний розмір об’єкта ≈ фізичний розмір/GSD. Для малих об’єктів потрібна краща деталізація або більший вхід."),
        ("8. Плитки","Великі сцени нарізають із перекриттям, координати прогнозів зсувають у систему кадру і зливають NMS."),
        ("9. PR, AP, mAP","PR-криву будують за відсортованою впевненістю; AP інтегрує precision за recall. mAP@[.5:.95] усереднює також за IoU 0,5…0,95."),
        ("10. AP за класом і розміром","Ці зрізи виявляють слабкі класи й малі об’єкти. Робочий поріг добирають на validation з урахуванням ціни FP/FN."),
        ("11. Категорії помилок","Клас, локалізація, обидва, дублікат, фон і пропуск розрізняють за IoU та відповідністю мітки."),
        ("12. Заходи","Пропуски малих об’єктів зменшують деталізацією й аугментацією; локалізацію — точнішою розміткою; фон — негативними прикладами й порогом.")]
    for q,a in qa:
        p=doc.add_paragraph();p.add_run(q).bold=True;doc.add_paragraph(a)
    doc.add_heading("10. Висновок",level=1)
    doc.add_paragraph(f"Побудовано детектор двох типів куп на аерознімках із розділенням за локаціями. На test отримано mAP@0,5={num(fine['map50'])}, mAP@[0,5:0,95]={num(fine['map50_95'])}; контрольна модель дала {num(base['map50_95'])}. Зменшення ×2 змінило mAP до {num(x2['map50_95'])}, плитки 320 px — до {num(tiles['map50_95'])}. Результати обмежені 640 px фрагментами та відсутністю фізичного GSD.")
    doc.add_heading("Джерела",level=1)
    doc.add_paragraph("DroneWaste v1.0, Zenodo, https://zenodo.org/records/17045559; Morandini et al., Scientific Data 13 (2026), 1666, https://doi.org/10.1038/s41597-026-07970-1")
    doc.add_paragraph("Методичні вказівки: «Лабораторна робота № 04», локальний PDF; Torchvision Faster R-CNN: https://pytorch.org/vision/stable/models/faster_rcnn.html")
    style_report(doc, 4)
    output=ROOT/"Лаб4_Бондар_Аерокосмічні_купи.docx";doc.save(output)
    print(f"Saved {output}")


if __name__=="__main__":main()
