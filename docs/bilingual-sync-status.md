# وضعیت هم‌گامی نسخه‌های انگلیسی و فارسی

تاریخچهٔ گیت نشان نمی‌دهد نسخهٔ فارسیِ یک صفحه از متن انگلیسی‌اش عقب افتاده است یا نه: دو فایل
جدا با دو تاریخ کامیت متفاوت‌اند و هیچ‌جا نوشته نمی‌شود کدام ترجمه مربوط به کدام نسخهٔ انگلیسی است.
این فایل همان دفترِ ثبت است. هر بار که متن انگلیسی صفحه‌ای تغییر می‌کند و ترجمه‌اش هنوز
به‌روز نشده، اینجا یادداشت می‌شود.

## چطور فهرست را تازه کنیم

```bash
python tools/check_bilingual_sync.py            # فقط ناهم‌گام‌ها و صفحه‌های بدون ترجمه
python tools/check_bilingual_sync.py --list-all # همهٔ صفحه‌ها، شامل هم‌گام‌ها
```

اسکریپت ساختار دو نسخه را مقایسه می‌کند — نه متن را، چون ترجمه طبیعتاً متفاوت است:

| سنجه | چرا |
| --- | --- |
| شمار سلول‌های `markdown` | بخش تازه‌ای به یک طرف اضافه شده |
| شمار سلول‌های `code` | مثال یا آزمایش تازه‌ای اضافه شده |
| متن سلول‌های `code` | قاعدهٔ `docs/bilingual.md` بند ۳: کد بایت‌به‌بایت کپی می‌شود |
| شمار سرفصل‌ها | ساختار بخش‌ها یکی نیست |
| شمار ارجاع‌های `img/` | تصویری اضافه/حذف شده |

این مقایسه **جایگزین خواندن دو متن نیست**؛ اگر ترجمه‌ای محتوایش کهنه باشد ولی شمار سلول‌ها و
سرفصل‌ها یکی باشد، این اسکریپت آن را نمی‌بیند. سنجهٔ مطمئن‌تر: تاریخ آخرین ویرایش معنادار متن
انگلیسی را کنار نام صفحه بنویسید.

## ناهم‌گام (بازبینی ۲۴ آبان ۱۴۰۵ / ۱۵ نوامبر ۲۰۲۶)

| صفحهٔ انگلیسی | نسخهٔ فارسی | تفاوت | یادداشت |
| --- | --- | --- | --- |
| `notebooks/Clustering-Validation.ipynb` | `fa/Clustering-Validation.ipynb` | ۴۰ سلول markdown در برابر ۴۱، ۲۲ سلول کد در برابر ۲۱، ۵۵ سرفصل در برابر ۵۰، ۱ تصویر در برابر ۳ | **۱۵ نوامبر ۲۰۲۶:** دو مثال شش‌نقطه‌ای (شمارش جفتی TP/FP/FN/TN و جدول Jaccard/Rand/ARI) و بخش تازهٔ «Choosing the Number of Clusters» (روش زانو با معیار تغییر شیب، نسخهٔ وتری Kneedle، و یک نمودار SSE با جدول بهره‌ها) به نسخهٔ انگلیسی افزوده شد؛ NMI، gap statistic و بخش Assignment Problem از این صفحه بیرون رفتند (به `Appendix-Clustering-Measures-and-Variants.ipynb`، که k-means++ و mini-batch k-means را هم دارد). ترجمهٔ فارسی هیچ‌کدام از این تغییرها را ندارد. |
| `notebooks/Kernel-Trick.md` | `fa/Kernel-Trick.md` | ۱۴۶ بلوک در برابر ۱۴۸ | بخش Exercises و Sources در نسخهٔ انگلیسی از مخزن MFDS آمده؛ فارسی عقب است. |
| `notebooks/MLE-Introduction.md` | `fa/MLE-Introduction.md` | ۸۹ بلوک در برابر ۹۰ | یک بخش اضافه در یکی از دو نسخه. |
| `notebooks/Appendix-Why-k-Means-Cannot-Be-Solved-by-Differentiation-Alone.md` | `fa/Appendix-Why-k-Means-Cannot-Be-Solved-by-Differentiation-Alone.md` | ۸۰ بلوک در برابر ۸۲ | — |

سه تغییر ساختاریِ ۱۵ نوامبر ۲۰۲۶ در **هر دو** نسخه یک‌سان اعمال شد، پس ناهم‌گامی محسوب نمی‌شود
و لازم نیست در جدول بالا ردیفی بگیرد:

- H1 هر صفحه اولین بلوک شد و سوییچ زبان زیر آن رفت (پیش از این سوییچ/بنر بالای H1 بود و عنوان
  صفحه دو بار چاپ می‌شد)؛
- تصویر `img/banner.png` از همهٔ صفحه‌ها برداشته شد و فقط در `index.md` هر دو نسخه ماند؛
- فرمول‌های `Appendix-Why-k-Means…` از `\(...\)`/`\[...\]` به `$...$`/`$$...$$` تبدیل شد.

## صفحه‌هایی که نسخهٔ فارسی ندارند (۳۰ صفحه)

`notebooks/Advanced-Topics.md`، `Appendix-Bayesian-Learning-Gaussian-Processes.ipynb`،
`Appendix-BDT-Discrete-Features.md`، `Appendix-Clustering-Measures-and-Variants.ipynb`،
`Appendix-Kernel-SVM-and-Kernel-Regression.ipynb`،
`Appendix-Sequence-Data.md`، `Appendix-Support-Vector-Machines.ipynb`،
`Bias-Variance-Tradeoff.ipynb`، `Data-Preprocessing.ipynb`، `Decision-Trees.ipynb`،
`Ensemble-Learning.ipynb`، `Hidden-Markov-Models.ipynb`، `Learning-Theory-and-VC.md`،
`Learning-to-Rank.md`، `Linear-Models-Advanced-Topics.ipynb`، `Model-Evaluation.ipynb`،
`Project-and-Review.md`، `HWs/misc/{final-project, human-activity-exercises,
human-activity-exercises-en, practical-homeworks, practical-homeworks-en, Problems-Chapter-BDT,
Normal-Eq-vs-GD}.md/ipynb`، `course-fall-2026/course-plan-fall-2026.md`،
`course-fall-2026/project/final-project.md`، و سه صفحهٔ از پیش فارسی
`course-fall-2026/{student-one-pager-fa, presentation-topics-en-fa, seminar-framing-and-closing-fa}.md`
و `course-fall-2026/project/project-clinic-pt1-pt2-fa.md` (این چهار مورد به نسخهٔ **انگلیسی**
نیاز دارند، نه فارسی).

## ۱۹ صفحهٔ هم‌گام

بقیهٔ صفحه‌های دوزبانه از نظر ساختاری یکی‌اند؛ فهرستشان با `--list-all`.

## قاعدهٔ نگهداری

1. پیش از هر تغییر در متن انگلیسی یک صفحهٔ دوزبانه، به این فایل نگاه کنید.
2. اگر ترجمه را همان‌جا به‌روز نکردید، صفحه را به جدول بالا اضافه کنید (تاریخ + چه چیزی اضافه شد).
3. وقتی فارسی را به‌روز کردید، ردیف را حذف کنید و اگر شمارش‌ها یکی شد، اسکریپت خودش آن را
   هم‌گام گزارش می‌کند.
