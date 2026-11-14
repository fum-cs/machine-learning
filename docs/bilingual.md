# دوزبانه کردن مطالب درس یادگیری ماشین — دستور کار

این فایل دستور کارِ عامل هوشمند (یا انسان) است برای ترجمهٔ فارسیِ بقیهٔ مطالب این مخزن. قاعده‌ها را
عوض نکنید مگر اینکه شکستنشان را با build آزمایش کرده باشید.

همین الگو در مخزن درس «مبانی ریاضی علوم داده» (`fum-cs/mfds`) پیاده و با ۲۹ صفحهٔ فارسی آزمایش
شده است؛ این فایل همان روش را برای ساختار این مخزن بازنویسی می‌کند.

## ۰. وضعیت فعلی در یک نگاه

| چیز | وضعیت |
| --- | --- |
| `fa/myst.yml` | ساخته شده (پروژهٔ دوم MyST؛ بدون بخش `exports`) |
| `fa/index.md` | **ترجمه شده** — نمونهٔ مرجع یک صفحهٔ `.md` |
| `tools/build_sites.py` | کپی شده؛ هر دو نسخه را می‌سازد و سوییچ را وصل می‌کند |
| `tools/sync_fa_toc.py` | کپی و برای عنوان گروه‌های همین مخزن تنظیم شده |
| `css/fa.css` | کپی شده؛ `fa/myst.yml` تنها جایی است که به آن اشاره می‌کند |
| `fa/references.bib` | کپی `notebooks/references.bib` (پروژهٔ فارسی به بیرون `fa/` دسترسی ندارد) |
| `fa/img/fum-cs-logo.png`, `fa/img/fum-logo.ico` | کپی تصاویر لوگو برای `site.options` |
| `fa/Clustering-Validation.ipynb`, `fa/Clustering-Part2.ipynb`, `fa/High-Dimensional-Data.ipynb` | **ترجمه شده** — نمونهٔ مرجع یک صفحهٔ `.ipynb` |
| `fa/index.md` و 16 صفحهٔ دیگر | **ترجمه شده** — نمونه‌ها: `.md`، `.ipynb` و پیوست‌ها |
| بقیهٔ 23 صفحه | **ترجمه نشده** — فهرست کامل در بخش ۱۳ |

هر صفحهٔ تازه‌ای که می‌سازید باید پس از آن `tools/sync_fa_toc.py` اجرا شود تا مدخلش در فهرست
فارسی فعال شود. سرصفحهٔ سوییچ در اولین سلول markdown نوتب‌وک قرار می‌گیرد.

## ۱. معماری: دو پروژهٔ MyST در یک مخزن

| مسیر | چیست |
| --- | --- |
| `myst.yml` + `notebooks/` + `course-fall-2026/` | نسخهٔ انگلیسی؛ مرجع و منبع حقیقت (canonical) |
| `fa/myst.yml` + `fa/` | نسخهٔ فارسی؛ همان ساختار و **همان نام فایل** |
| `css/fa.css` | RTL برای فارسی؛ کد، فرمول و جدول را به LTR برمی‌گرداند |
| `css/rtl.css` | روش قدیمی تک‌پروژه‌ای (بند ۱۲ را ببینید) — با این روش کاری نکنید |
| `tools/build_sites.py` | هر دو نسخه را می‌سازد، فارسی را در `_build/html/fa` می‌گذارد |
| `tools/sync_fa_toc.py` | فهرست فارسی را از فهرست انگلیسی بازتولید می‌کند |

قاعدهٔ مسیر: پیشوند `notebooks/` در نسخهٔ فارسی **حذف** می‌شود، بقیهٔ مسیر دست نمی‌خورد:

| انگلیسی | فارسی |
| --- | --- |
| `notebooks/index.md` | `fa/index.md` |
| `notebooks/Clustering-Part1.ipynb` | `fa/Clustering-Part1.ipynb` |
| `notebooks/Appendix-Sequence-Data.md` | `fa/Appendix-Sequence-Data.md` |
| `course-fall-2026/project/final-project.md` | `fa/course-fall-2026/project/final-project.md` |

**نام فایل و جای آن در درخت پوشه‌ها هرگز ترجمه نمی‌شود**؛ همین تقارن است که سوییچ زبان و لینک‌های
نسبی را قابل محاسبه می‌کند.

## ۲. گام‌به‌گام برای یک صفحهٔ `.md`

1. متن انگلیسی را کامل بخوانید (نه فقط عنوان).
2. فایل فارسی را در مسیر متناظر زیر `fa/` بسازید و متن را ترجمه کنید.
3. این بلوک را **اولین** چیزی در فایل بگذارید (برای وصل شدن به نسخهٔ انگلیسی):

   ```html
   <div class="langswitch">
     <a href="#lang-en">English</a> · <strong>فارسی</strong>
   </div>
   ```

   `href="#lang-en"` جای‌نگهدار است؛ `tools/build_sites.py` بعد از build آن را به مسیر نسبی
   درست به نسخهٔ انگلیسی همان صفحه تبدیل می‌کند.

4. در فایل **انگلیسی** هم همین بلوک را با `langswitch-ltr` و `#lang-fa` (فقط یک بار) بگذارید:

   ```html
   <div class="langswitch langswitch-ltr">
     <strong>English</strong> · <a href="#lang-fa">فارسی</a>
   </div>
   ```

5. تصاویر: هر مسیر تصویرِ متن انگلیسی (`img/...`) باید **در درخت `fa/` هم وجود داشته باشد** — فایل
   را در `fa/img/` کپی کنید و مسیر را در متن دست نزنید.
6. `python tools/sync_fa_toc.py` را اجرا کنید (مدخل خودکار فعال می‌شود).

## ۳. گام‌به‌گام برای یک نوتبوک `.ipynb`

قاعده: **فقط سلول‌های `markdown` ترجمه می‌شوند**. سلول‌های `code` با خروجی‌ها و
`execution_count` دست‌نخورده کپی می‌شوند تا دو نسخه هرگز در نتیجه‌ها واگرا نشوند و لازم نباشد چیزی
اجرا شود.

### ۳.۱ فهرست سلول‌ها را ببینید

```python
import json
nb = json.load(open("notebooks/PCA-Part2.ipynb"))
for i, c in enumerate(nb["cells"]):
    print(i, c["cell_type"], "".join(c["source"])[:70].replace("\n", " "))
```

شماره‌گذاری از صفر است و شامل سلول‌های کد هم می‌شود.

### ۳.۲ فایل فارسی را بنویسید

`nbformat.write` کل فایل را بازقالب‌بندی می‌کند (فهرست خط‌های `source` را به یک رشته تبدیل
می‌کند) و diff را شلوغ می‌کند. برای فایل **فارسی** این اهمیتی ندارد، ولی همان‌جا هم بهتر است با
`json` کار کنید تا سبک فایل انگلیسی حفظ شود:

```python
import json, pathlib

SRC = pathlib.Path("notebooks/PCA-Part2.ipynb")
DST = pathlib.Path("fa/PCA-Part2.ipynb")
FA = {0: "...", 1: "...", 3: "..."}          # شمارهٔ سلول -> متن فارسی

raw = SRC.read_text(encoding="utf-8")
indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip()) or 1
nb = json.loads(raw)

md_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "markdown"]
assert not [i for i in md_idx if i not in FA], "سلول ترجمه‌نشده"
assert not [i for i in FA if nb["cells"][i]["cell_type"] != "markdown"], "سلول غیر markdown"

for i, text in FA.items():
    nb["cells"][i]["source"] = text.splitlines(keepends=True)

SW = ('<div class="langswitch">\n'
      '  <a href="#lang-en">English</a> · <strong>فارسی</strong>\n'
      '</div>\n')
nb["cells"][0]["source"] = (SW + "\n" + "".join(nb["cells"][0]["source"])).splitlines(keepends=True)

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=indent, ensure_ascii=False)
```

دو `assert` بالا مهم‌اند: یکی تضمین می‌کند هیچ سلول markdown جا نیفتاده، دیگری که اشتباهی متن
فارسی روی سلول کد ننشسته باشد.

### ۳.۳ سوییچ را به نسخهٔ انگلیسی اضافه کنید

بدون `nbformat`، چهار سطر رشته را در **اولین** آرایهٔ `source` فایل انگلیسی وارد کنید:

```python
import json, re, pathlib

p = pathlib.Path("notebooks/PCA-Part2.ipynb")
s = p.read_text(encoding="utf-8")
if "langswitch" not in s:
    lines = ['<div class=\\"langswitch langswitch-ltr\\">\\n',
             '  <strong>English</strong> · <a href=\\"#lang-fa\\">فارسی</a>\\n',
             '</div>\\n', '\\n']
    m = re.search(r'"source": \[\n( *)"', s)
    lead = m.group(1)
    block = "".join(f'{lead}"{line}",\n' for line in lines)
    i = m.start(0) + len('"source": [\n')
    s = s[:i] + block + s[i:]
    json.loads(s)                      # اعتبارسنجی پیش از نوشتن
    p.write_text(s, encoding="utf-8")
```

## ۴. چیزهایی که ترجمه نمی‌شوند

- کد، نام متغیرها، نام کلاس‌ها و توابع، کامنت‌های داخل کد (اختیاری، ولی بهتر است دست نخورد)،
  خروجی‌ها، `execution_count`.
- فرمول‌های `$...$` و `$$...$$`، `\ref`/`\label`، `{numref}`، شناسه‌های headingها،
  `{figure}`/`{table}`.
- مسیر فایل‌ها، لینک‌های بیرونی، `title:` در `myst.yml`، نام فایل، عنوان گروه‌ها در `myst.yml`
  انگلیسی.
- فهرست `exports` (PDF/TeX) — پروژهٔ فارسی آن را ندارد و نباید داشته باشد.

## ۵. لینک‌های داخلی

لینک‌ها در فارسی باید **به صفحهٔ فارسی** اشاره کنند:

- `[k-means](Clustering-Part1.ipynb)` (نه `../notebooks/...`)؛
- برای صفحه‌های بیرون `notebooks/`: `[پروژه](course-fall-2026/project/final-project.md)`.

هر صفحه‌ای که در نسخهٔ انگلیسی منتشر نشده (در `myst.yml` کامنت است یا فایلش نیست) فارسی هم
نمی‌شود، تا سوییچ به صفحهٔ ناموجود اشاره نکند.

> **توجه:** مدخل `notebooks/Appendix-Why-k-Means-Cannot-Be-Solved-by-Differentiation-Alone.md` در
> فهرست انگلیسی هست ولی **فایلش در این مخزن نیست**. تا آن فایل افزوده نشود، build انگلیسی هم برای
> آن هشدار می‌دهد و ترجمه‌اش معنا ندارد.

## ۶. تصاویر و منابع

- تصویرها: `img/...` در متن انگلیسی یعنی `fa/img/...` در فارسی. فقط تصویرهایی را کپی کنید که همان
  صفحه ارجاع می‌دهد، نه کل پوشه را.
- کتاب‌نامه: صفحه‌های دارای `{cite}` به `bibliography` پروژه نیاز دارند. `fa/references.bib` کپی
  `notebooks/references.bib` است و در `fa/myst.yml` با مسیر `references.bib` ثبت شده. اگر
  `references.bib` انگلیسی به‌روز شد، کپی فارسی را هم به‌روز کنید.

## ۷. اصطلاح‌نامه (یکدست بمانید)

| English | فارسی |
| --- | --- |
| data point / sample | نقطهٔ داده / نمونه |
| feature / dimension | ویژگی / بُعد |
| target / response | متغیر هدف / پاسخ |
| label | برچسب |
| training / validation / test set | مجموعهٔ آموزش / اعتبارسنجی / آزمون |
| supervised / unsupervised / semi-supervised | با نظارت / بی‌نظارت / نیمه‌نظارتی |
| classification / regression / ranking | دسته‌بندی / رگرسیون / رتبه‌بندی |
| clustering / cluster | خوشه‌بندی / خوشه |
| centroid | مرکز (سنتروئید) |
| partition | افراز |
| hypothesis / estimator | فرضیه / برآوردگر |
| loss / cost function | تابع هزینه |
| objective | تابع هدف |
| gradient descent | گرادیان کاهشی |
| learning rate | نرخ یادگیری |
| overfitting / underfitting | بیش‌برازش / کم‌برازش |
| regularization | تنظیم‌کننده (رگولاریزاسیون) |
| bias–variance tradeoff | مبادلهٔ بایاس–واریانس |
| cross-validation | اعتبارسنجی متقاطع |
| confusion matrix | ماتریس درهم‌ریختگی |
| precision / recall / F1 | دقت / بازآوری / F1 |
| likelihood / log-likelihood | درست‌نمایی / لگاریتم درست‌نمایی |
| maximum likelihood estimation (MLE) | برآورد بیشینهٔ درست‌نمایی |
| prior / posterior | پیشین / پسین |
| expectation–maximization (EM) | امید–بیشینه‌سازی (EM) |
| latent variable | متغیر پنهان |
| mixture model | مدل مخلوط |
| kernel / kernel trick | هسته / شگرد هسته |
| margin / support vector | حاشیه / بردار پشتیبان |
| decision boundary | مرز تصمیم |
| eigenvector / eigenvalue | بردار ویژه / مقدار ویژه |
| singular value | مقدار تکین |
| rank | رتبه |
| subspace | زیرفضا |
| projection | تصویر (تصویرسازی) |
| principal component | مؤلفهٔ اصلی |
| dimensionality reduction | کاهش بُعد |
| curse of dimensionality | **نفرین ابعاد** (نه «نفرین بُعد») |
| high-dimensional | **با ابعاد بالا** (نه «بُعد بالا») |
| hypersphere | **اَبَرکره** (نه «فراکره») |
| low-rank | **رتبه پایین** (نه «کم‌مرتبه») |
| derivation | **استنتاج** (نه «اشتقاق») |
| prototype | پیش‌نمونه |
| quantization | چنده‌گذاری |
| contingency table | جدول توافقی |
| purity / silhouette | خلوص / سیلوئت |
| ground truth | حقیقت‌مبنا |
| dendrogram / linkage | دندروگرام / پیوند |
| exercise / seminar | تمرین / سمینار |
| quiz / final exam | کوییز / آزمون پایان ترم |

اصطلاح تازه را همین جدول اضافه کنید تا نسخه‌های بعدی یکدست بمانند. پنج اصطلاح پررنگ
(**ابَرکره**، **با ابعاد بالا**، **نفرین ابعاد**، **رتبه پایین**، **استنتاج**) بازبینی‌شده‌اند و
جایگزین‌هایشان («فراکره»، «بُعد بالا»، «نفرین بُعد»، «کم‌مرتبه»، «اشتقاق») نباید به کار روند.

## ۸. ساخت و بررسی

```bash
# هر دو نسخه:
python tools/build_sites.py
# فقط فارسی:
python tools/build_sites.py fa
```

روی ویندوز/git-bash (چون MSYS مقدار `/fa` را به مسیر ویندوزی تبدیل می‌کند):

```bash
MSYS_NO_PATHCONV=1 JB='C:/programs/anaconda3/envs/pth/Scripts/jupyter-book.exe' \
  python tools/build_sites.py
```

اگر Jupyter Book 2 روی `PATH` نیست، مسیر اجرایی‌اش را در `JB` بگذارید. متغیر `PORT` عمداً از
محیط حذف می‌شود؛ MyST آن را به‌عنوان پورت سرور تفسیر می‌کند و build را می‌شکند.

بعد از build دست‌کم این چهار چیز را چک کنید:

1. `echo $?` صفر باشد و در لاگ `[switch]` تعداد صفحه‌های فارسی را درست بشمارد.
2. `_build/html/fa/<slug>/index.html` وجود داشته باشد و در آن `dir="rtl"` باشد.
3. لینک سوییچ در صفحهٔ فارسی به **همان صفحهٔ انگلیسی** برود، نه به صفحهٔ اصلی:

   ```bash
   grep -o '<div class="langswitch">.\{0,80\}' _build/html/fa/index.html
   ```

4. تصویرها با پیشوند `/fa/` بار شوند:

   ```bash
   grep -o 'src="[^"]*_public[^"]*"' _build/html/fa/pca-part1/index.html | head
   ```

**slug صفحه‌ها کوچک‌شدهٔ نام فایل است**: `PCA-Part1.ipynb` → `/fa/pca-part1/`، نه
`/fa/PCA-Part1/`. هنگام بررسی با `curl` همین را به کار ببرید.

برای دیدن سایت: در `_build/html` یک سرور ساده بالا بیاورید و `/` و `/fa/` را باز کنید:

```bash
python3 -m http.server 8140 --bind 127.0.0.1
```

پیش از build سرور را ببندید وگرنه (روی ویندوز) `EPERM` می‌گیرید.

## ۹. تله‌های شناخته‌شده

- **فرمول‌ها با KaTeX رندر می‌شوند**، نه MathJax: `css/fa.css` باید `.katex` و `.katex-display`
  را به LTR برگرداند. اگر فقط `mjx-container` را هدف بگیرید، فرمول درون‌خطی راست‌به‌چپ آینه
  می‌شود.
- **شمارهٔ فرمول** باید منتهی‌الیه راست بماند یا دست‌کم فاصله داشته باشد؛ قاعدهٔ `gap` و `direction:
  ltr` روی ردیف نمایش فرمول در `css/fa.css` همین کار را می‌کند.
- **داخل HTML خام، مارک‌داون کار نمی‌کند.** در چیپ سوییچ باید `<strong>فارسی</strong>` بنویسید،
  نه `**فارسی**`.
- **`nbformat.write` فایل را بازقالب‌بندی می‌کند**؛ برای نسخهٔ انگلیسی از همان روش خام JSON بند
  ۳.۳ استفاده کنید.
- تگ `<script>` و attributeهای `data-*` در HTML خام **حذف می‌شوند**؛ برای همین سوییچ با
  جای‌نگهدار `#lang-…` و بازنویسی در زمان build ساخته شده است.
- **build انگلیسی `_build/html/fa` را پاک می‌کند.** `tools/build_sites.py` پس از هر build خروجی
  فارسی را برمی‌گرداند، ولی `jupyter-book build --html` دستی این حفاظت را ندارد.
- **`{include}` کار نمی‌کند** (متن خام چاپ می‌شود).
- MyST خودش `href`های داخلی را بازنویسی می‌کند؛ هیچ لینکی را با مسیر مطلق دستی ننویسید.
- در صفحهٔ اصلی، عنوان از `myst.yml` می‌آید و H1 صفحه زیر آن نشان داده می‌شود؛ H1 را حذف نکنید.
- برچسب‌های خود قالب (Contents, Search, Exercise) انگلیسی می‌مانند؛ MyST هنوز ترجمهٔ رابط ندارد.
- **روش قدیمی `css/rtl.css`** که در این مخزن هست (هر صفحه را در `:::{div}` با کلاس `rtl` می‌پیچد)
  با روش دوپروژه‌ای جمع نمی‌شود: متنِ درون HTML خام را MyST پارس نمی‌کند، کل صفحه RTL نمی‌شود و
  نگهداری‌اش سخت است. از آن استفاده نکنید؛ `fa/myst.yml` به `css/fa.css` اشاره می‌کند.
- `notebooks/_build/` (خروجی قدیمی داخل پوشهٔ نوتبوک‌ها) را در جست‌وجوها و کپی‌ها نادیده بگیرید.

## ۱۰. انتشار

workflow فعلی (`.github/workflows/deploy.yml`) هر دو نسخه را با `python3 tools/build_sites.py`
می‌سازد و پس از آن وجود `_build/html/fa/index.html` و `dir="rtl"` را بررسی می‌کند، پس سایت
منتشرشده نسخهٔ فارسی را دارد. اگر گام ساخت به `jupyter-book build --html` برگردد، فارسی از
آرتیفکت حذف می‌شود. گام ساخت چنین است:

```yaml
      - name: Build both editions
        run: python tools/build_sites.py
```

و مطمئن شوید `BASE_URL` همان‌طور که workflow تنظیم می‌کند به `build_sites.py` می‌رسد (اسکریپت
`en_base` و `fa_base` را از آن می‌سازد). پیش از هر push، build محلی را با همان `BASE_URL`
آزمایش کنید.

## ۱۱. تعریف «این صفحه تمام شد»

یک صفحه وقتی تمام است که:

1. فایل فارسی در مسیر متناظر `fa/` باشد و همهٔ سلول‌های markdown ترجمه شده باشند؛
2. سوییچ در **هر دو** نسخه باشد؛
3. لینک‌های داخلی به صفحه‌های فارسی اشاره کنند و تصویرهای ارجاع‌شده در `fa/img/` کپی شده باشند؛
4. `python tools/sync_fa_toc.py` اجرا شده و مدخلش در `fa/myst.yml` فعال است؛
5. `python tools/build_sites.py` با صفر warning/error تمام شود و شمارش `[switch]` یکی زیاد شده
   باشد؛
6. اصطلاح‌های بند ۷ رعایت شده باشند (جست‌وجوی پنج واژهٔ مردود باید صفر نتیجه بدهد).

## ۱۲. ترتیب پیشنهادی ترجمه

به ترتیب خود درس (فهرست `myst.yml`):

1. **مبانی**: `Introduction`، `Local-Models-and-kNN`، `Clustering-Part1`،
   `Linear-Regression-Part1`
2. **روش‌های ساده و داده‌های با ابعاد بالا**: `Clustering-Part2`، `Clustering-Validation`،
   `High-Dimensional-Data`
3. **مدل‌های خطی و شبکه‌های عصبی**: `Linear-Regression-Part2`، `MLP-and-Backpropagation`
4. **کاهش بُعد**: `PCA-Part1`، `PCA-Part2`، `PCA-and-LDA`
5. **روش‌های هسته**: `Feature-Map`، `Kernel-Trick`، `Kernel-Regression`
6. **یادگیری بیزی و آماری**: `MLE-Introduction`، `Naive-Bayes`، `Mahalanobis-Distance`،
   `Bayesian-Decision-Theory`
7. **مدل‌های متغیر پنهان**: `Gaussian-Mixtures-Part1`، `Gaussian-Mixtures-Part2`
8. **موضوع‌های دیگر**: `Bias-Variance-Tradeoff`، `Data-Preprocessing`، `Model-Evaluation`،
   `Decision-Trees`، `Ensemble-Learning`، `Hidden-Markov-Models`، `Learning-to-Rank`،
   `Learning-Theory-and-VC`، `Advanced-Topics`
9. **برنامهٔ درس و پروژه**: `course-fall-2026/*`
10. **پیوست**: `Appendix-BDT-Discrete-Features`، `Appendix-Sequence-Data`،
    `Linear-Models-Advanced-Topics`

## ۱۳. فهرست کامل صفحه‌های باقی‌مانده (23 صفحه)

`MLE-Introduction.md`، `Naive-Bayes.ipynb`،
`Mahalanobis-Distance.ipynb`، `Bayesian-Decision-Theory.ipynb`،
`Gaussian-Mixtures-Part1.ipynb`، `Gaussian-Mixtures-Part2.ipynb`،
`Bias-Variance-Tradeoff.ipynb`، `Data-Preprocessing.ipynb`،
`Model-Evaluation.ipynb`، `Decision-Trees.ipynb`،
`Ensemble-Learning.ipynb`، `Hidden-Markov-Models.ipynb`،
`Learning-to-Rank.md`، `Learning-Theory-and-VC.md`،
`Advanced-Topics.md`، `course-fall-2026/course-plan-fall-2026.md`،
`course-fall-2026/project/final-project.md`، `course-fall-2026/student-one-pager-fa.md`،
`course-fall-2026/presentation-topics-en-fa.md`، `course-fall-2026/seminar-framing-and-closing-fa.md`،
`Appendix-BDT-Discrete-Features.md`، `Appendix-Sequence-Data.md`،
`Linear-Models-Advanced-Topics.ipynb`

سه صفحهٔ `course-fall-2026/student-one-pager-fa.md`، `course-fall-2026/presentation-topics-en-fa.md` و `course-fall-2026/seminar-framing-and-closing-fa.md` از پیش فارسی‌اند و برای دوزبانه‌شدن به نسخهٔ **انگلیسی** نیاز دارند، نه فارسی.

## ۱۴. محتوای مشترک با درس «مبانی ریاضی علوم داده»

برخی صفحه‌های این دو درس عملاً یکی‌اند و ترجمهٔ فارسی‌شان قابل بازاستفاده است:

| صفحه | وضعیت |
| --- | --- |
| `notebooks/MLP-and-Backpropagation.ipynb` | **بایت‌به‌بایت یکسان** با نسخهٔ مخزن MFDS |
| `notebooks/Kernel-Trick.md` | تقریباً یکسان (تفاوت چند خطی) |
| `Clustering-Part1`، `Clustering-Part2`، `Clustering-Validation`، `Bias-Variance-Tradeoff`، `PCA-Part1`، `PCA-Part2`، `Kernel-Regression` | هم‌نام ولی **محتوای متفاوت** — ترجمه را از همین مخزن انجام دهید |

در مخزن MFDS مسیر صفحه‌ها زیرپوشه‌ای است (`notebooks/high-dimensional-spaces/…`،
`notebooks/learning/…`) ولی ترجمه‌ها با همان نام فایل زیر `fa/` هستند؛ اگر صفحه‌ای آنجا ترجمه شده
بود، می‌توانید فایل فارسی‌اش را بردارید، مسیر/لینک‌های داخلی را با ساختار این مخزن هماهنگ کنید
(و `img`ها را کپی کنید) — اما پیش از استفاده، تفاوت متن انگلیسی دو نسخه را ببینید.

## ۱۵. نمونهٔ مرجع

`fa/index.md` نمونهٔ کامل یک صفحهٔ دوزبانه است: ترجمهٔ کامل متن، سوییچ دوطرفه، مدخل فعال در
`fa/myst.yml` و لینک‌های داخلی به صفحه‌های فارسی. برای صفحهٔ `.md` بعدی همین را الگو بگیرید.
