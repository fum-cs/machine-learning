# دستور کار بازبینی: «Simple Methods and High Dimension»

> مخزن: `C:\git\fum-cs\machine-learning` · شاخه: `main` · تاریخ تهیه: ۱۰ نوامبر ۲۰۲۶
> این فایل نتیجهٔ یک بازبینی است؛ **هنگام نوشتن آن هیچ اصلاحی در نوتب‌وک‌ها انجام نشده است.**
> خطاب این سند یک عامل هوشمند است که اصلاحات را اجرا می‌کند. متن نوتب‌وک‌ها انگلیسی است؛
> همهٔ اصلاحات متنی داخل نوتب‌وک‌ها باید **انگلیسی** و هم‌سبک با متن موجود باشند.

---

## ۰. دامنه

چهار فایل زیر، که در `myst.yml` زیر عنوان «Simple Methods and High Dimension» فهرست شده‌اند:

| # | فایل | سلول | متن (کاراکتر) | کد | جایگاه در درس |
|---|------|------|---------------|-----|----------------|
| ۱ | `notebooks/Clustering-Part2.ipynb` | ۱۱ | ۶٬۶۰۰ | ۴ | M1 — k-means در عمل (۹۰′) |
| ۲ | `notebooks/Clustering-Validation.ipynb` | ۶۳ | ۳۰٬۵۰۰ | ۲۳ | M2 — اعتبارسنجی خوشه‌بندی (۹۰′) |
| ۳ | `notebooks/High-Dimensional-Data.ipynb` | ۲۷ | ۱۷٬۶۰۰ | ۱۰ | M7 — بُعد بالا |
| ۴ | `notebooks/High-Dimensional-Data-and-kNN.ipynb` | ۴ | ۲۱۴ | ۲ | صفحهٔ مستقل (ناتمام) |

بلوک مربوط در `myst.yml` (حدود خط ۱۸۲):

```yaml
    - title: Simple Methods and High Dimension
      children:
        - file: "notebooks/Clustering-Part2.ipynb"
        - file: "notebooks/Clustering-Validation.ipynb"
        - file: "notebooks/High-Dimensional-Data.ipynb"
        - file: "notebooks/High-Dimensional-Data-and-kNN.ipynb"
```

### قواعد اجرا

1. **فقط** چهار فایل بالا (و در صورت لزوم بلوک TOC در `myst.yml`) تغییر کند. به `css/rtl.css`،
   `myst.yml` بیرون از این بلوک، `notebooks/references.bib` و بقیهٔ نوتب‌وک‌ها دست نزنید.
2. نوتب‌وک را **بازنویسی کلی نکنید**؛ تغییرات نقطه‌ای و قابل‌ردیابی باشند. شمارهٔ سلول‌های موجود
   در این سند مبنای کار است؛ اگر سلولی اضافه کردید، شماره‌ها جلو می‌روند — قبل از هر آیتم جدید
   ساختار فعلی فایل را دوباره بخوانید.
3. هر جایی که **کد** عوض شد، سلول‌ها باید دوباره اجرا شوند تا خروجی ذخیره‌شده با کد هم‌خوان باشد.
   دستور پیشنهادی (محیط باید `numpy matplotlib pandas scipy scikit-learn networkx seaborn`
   و در صورت نگه‌داشتن آن بخش `yellowbrick` را داشته باشد):
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/<file>.ipynb
   ```
   اگر محیط اجرا آماده نبود، تغییر کد را انجام ندهید و فقط در گزارش پایانی ثبت کنید؛
   خروجی کهنه و متناقض بدتر از نبود تغییر است.
4. سبک موجود حفظ شود: جمله‌های کوتاه، تأکید با `**...**` برای اصطلاح اول، ریاضی داخل `$...$` و
   `$$...$$`، بدون تیتر تزئینی و بدون ایموجی.
5. هر آیتم با شناسه (مثل `P0-3`) در فهرست کار علامت بزنید تا قابل پیگیری باشد.

---

## ۱. تصمیم‌های باز (پیش‌فرض پیشنهادی برای اجرا)

این‌ها تصمیم آموزشی‌اند، نه فنی. اگر استاد تصمیم دیگری نگرفت، پیش‌فرض را اجرا کنید و در گزارش
پایانی ذکر کنید کدام پیش‌فرض اعمال شده است.

| # | موضوع | پیش‌فرض پیشنهادی |
|---|-------|------------------|
| D1 | نماد تعداد خوشه/کلاس | `k` = تعداد خوشه، `d` = بُعد، `r` = تعداد کلاس (قرارداد سراسری درس) |
| D2 | NMI | تعریف و محاسبه شود (چهار ارجاع در متن هست ولی هیچ تعریفی نیست) |
| D3 | صفحهٔ kNN | در `High-Dimensional-Data` ادغام شود و از TOC حذف شود |
| D4 | وعدهٔ لم جانسون–لیندن‌شتراوس | فقط نرم شود (جملهٔ «در ادامهٔ درس» جای «bridge»)، صفحهٔ جدید ساخته نشود |
| D5 | k-means++ | به `Clustering-Part2` افزوده شود (برنامهٔ M1 صریحاً ۱۰′ برای آن دارد) |

---

## ۲. P0 — خطاهای درستی (اولویت اول، کم‌حجم و قطعی)

### P0-1 — `Clustering-Validation.ipynb`، سلول ۳۸: جمع جدول غلط است
جدول «Good Clustering Results» با `n = 100` بسته می‌شود، در حالی که جمع سطرها ۶۱+۵۰+۳۹ = ۱۵۰
و جمع ستون‌ها ۵۰+۵۰+۵۰ = ۱۵۰ است.
**تغییر:** `**n = 100**` ← `**n = 150**`.

### P0-2 — `Clustering-Validation.ipynb`، سلول ۳۹: مقدار Purity غلط است
متن فعلی:

```
**Purity:** 0.667
**Match:** 0.560
**F:** 0.658
```

بیشینهٔ هر سطر ۳۰+۴+۵۰ = ۸۴ است، پس purity = ۸۴/۱۵۰ = **۰٫۵۶**؛ عدد ۰٫۶۶۷ همان ۱۰۰/۱۵۰
است و از جایی دیگر مانده. (اینکه `Match` هم ۰٫۵۶ است تصادفی نیست: هر سطر یک بیشینهٔ یکتا دارد،
پس تطبیق بیشینه همان ۸۴ را برمی‌گرداند.)
**تغییر:** `**Purity:** 0.667` ← `**Purity:** 0.560`. و یک خط زیر جدول:
`Here purity happens to equal match because each cluster has a single unambiguous majority class.`

### P0-3 — `Clustering-Validation.ipynb`، سلول ۳۲: `max` به‌جای `argmax`
متن فعلی:

```
Given cluster $C_i$, let $j_i$ denote the partition that contains the maximum number of points from $C_i$,
that is, $j_i = \max_{j=1}^k \{ n_{ij} \}$.
```

این نوشتار می‌گوید اندیس کلاس برابر بیشترین شمارش است.
**تغییر:** `$j_i = \arg\max_{j} n_{ij}$`.

### P0-4 — `Clustering-Validation.ipynb`، سلول ۳: ترتیب اندیس‌ها در نسخهٔ $O(n)$
متن فعلی: «incrementing the corresponding count $n_{y_i\hat{y}_i}$».
با تعریف $\mathbf{N}(i,j) = |C_i \cap T_j|$ (سطر = خوشه، ستون = کلاس)، ترتیب باید برعکس باشد.
**تغییر:** `$n_{y_i\hat{y}_i}$` ← `$n_{\hat{y}_i,\,y_i}$` و در ادامه جمله یک نیم‌خط توضیح:
`the first index is the cluster label, the second the class label`.

### P0-5 — `Clustering-Part2.ipynb`، سلول ۷: ادعای ریاضی غلط
متن فعلی: «Now, let's use the Lemma 2 to prove that the centroid minimizes the sum of squared distances».
لم ۲ (اتحاد فواصل جفتی) کمینه‌سازی را ثابت نمی‌کند؛ آن نتیجهٔ ۱ است که از لم ۱ می‌آید.
**تغییر:** جمله به این معنا بازنویسی شود که لم ۲ صورت جفتی همان کمیت را می‌دهد و عددی
معادل $n\sum_i\|\mathbf{x}_i-\mu\|^2$ است، و تأیید می‌کند محاسبهٔ SSE از راه جفت‌ها و از راه
مرکزوار یکی است.

### P0-6 — `Clustering-Part2.ipynb`، سلول ۸: توصیف غلط لم ۲
متن فعلی: «The lemma states that the sum of squared distances of all data points is equal to the
number of data points times the squared distance from the centroid to the mean of the data points.»
این جمله بی‌معناست.
**تغییر:** توصیف دقیق: مجموع مربعات فاصلهٔ همهٔ جفت‌ها برابر است با $n$ ضرب‌در مجموع مربعات
فاصلهٔ نقاط تا مرکزوار؛ و اینکه تابع `lemma_2` هر دو سمت را عددی حساب و چاپ می‌کند.

### P0-7 — `Clustering-Part2.ipynb`، سلول ۱: پرانتز اضافه در اثبات لم ۲
خطوط فعلی:

```
&= \frac{1}{2} \left(\sum_{j=1}^n (\sum_{i=1}^n |x_i - \mu|^2) + n \sum_{j=1}^n(|\mu - x_j|^2)\right)\\
&= \frac{1}{2} \left(n\sum_{i=1}^n |x_i - \mu|^2) + n\sum_{i=1}^n |x_i - \mu|^2)\right)\\
```

خط دوم یک `)` اضافه بعد از `^2` دارد و `\left(`/`\right)` نامتوازن رندر می‌شود.
**تغییر:** خط دوم:
```
&= \frac{1}{2} \left(n\sum_{i=1}^n |x_i - \mu|^2 + n\sum_{i=1}^n |x_i - \mu|^2\right)\\
```

### P0-8 — `Clustering-Part2.ipynb`، سلول ۱: اثبات همگرایی وجود ندارد
متن فعلی: «To show convergence, we argue that the cost of the clustering … always improves.»
و بعد بلافاصله کد می‌آید؛ استدلال هرگز نوشته نشده. این ۲۰′ از جلسهٔ M1 است.
**تغییر:** افزودن ۵ تا ۸ خط (فرمول‌دار) با این ساختار:
1. در گام تخصیص، با مرکزوارهای ثابت، هر نقطه به نزدیک‌ترین مرکزوار می‌رود ⇒ SSE کاهش نمی‌یابد.
2. در گام به‌روزرسانی، با تخصیص ثابت، مرکزوار کمینه‌کنندهٔ SSE است (لم ۱ / نتیجهٔ ۱) ⇒ باز هم کاهش.
3. تعداد افرازهای متناهی از $n$ نقطه در $k$ خوشه متناهی است و SSE اکیداً کاهش می‌یابد ⇒ خاتمه.
4. تأکید: کمینهٔ محلی، نه سراسری.

### P0-9 — `Clustering-Part2.ipynb`، سلول ۱: «همیشه مثبت»
در نتیجهٔ ۱ نوشته شده «the second term … is always positive». عبارت $n\|\mu-z\|^2$ نامنفی است
و تنها در $z=\mu$ صفر می‌شود.
**تغییر:** «always positive» ← «non-negative, and zero only when $z = \mu$; hence the centroid is
the unique minimizer».

### P0-10 — `Clustering-Part2.ipynb`، سلول ۴ (و ۲): حلقهٔ بی‌پایان در حالت خوشهٔ خالی
شرط توقف `np.all(centroids == new_centroids)` است و `max_iter` وجود ندارد. اگر خوشه‌ای خالی شود،
`mean` روی آرایهٔ خالی `nan` می‌دهد و چون `nan == nan` نادرست است، حلقه هرگز تمام نمی‌شود.
**تغییر:**
- `while True` ← `for _ in range(max_iter)` با `max_iter=300`؛
- شرط توقف `np.allclose(centroids, new_centroids, atol=1e-9)`؛
- برای خوشهٔ خالی: مرکزوار قبلی حفظ شود (و در صورت تمایل یک `warnings.warn`)، تا `nan` تولید نشود؛
- اگر تابع تکراری `kmeans` (سلول ۲) حذف شد، همین اصلاح فقط در کلاس لازم است.

### P0-11 — `High-Dimensional-Data-and-kNN.ipynb`، سلول ۳: تیتر نمودار غلط و خروجی آلوده
- `plt.title("KMeans Clustering (2D) with True Labels")` در صفحه‌ای دربارهٔ kNN.
  **تغییر:** `"Two informative features with the true labels"`.
- `plt.legend()` بدون برچسب، هشدار `No artists with labels found to put in legend` را در خروجی
  ذخیره‌شده گذاشته است. **تغییر:** یا `label=` به هر `scatter` داده شود، یا `plt.legend()` حذف شود.
- ایمپورت‌های بی‌استفاده: `from scipy.stats import norm`، `seaborn`. **تغییر:** حذف.

### P0-12 — `%%HTML` / `require.js` در سلول اول
در `Clustering-Part2`، `High-Dimensional-Data` و `High-Dimensional-Data-and-kNN` سلول صفر با
`![](img/banner.png)` شروع می‌شود و بعد `%%HTML` و `<script src="require.js"></script>` دارد.
این بازماندهٔ Jupyter v1 است: در markdown سلول، دستور `%%HTML` اجرا نمی‌شود و به‌صورت متن در
صفحهٔ myst می‌آید و `<script>` هم به فایل ناموجود `require.js` اشاره می‌کند.
**تغییر:** در این سه فایل، دو خط `%%HTML` و `require.js` حذف و فقط بنر بماند. (این الگو در ۶۳
نوتب‌وک مخزن تکرار شده است؛ **در این دستور کار فقط این چهار فایل** — یعنی سه فایل بالا — اصلاح شود.)

---

## ۳. P1 — انسجام، ساختار و حجم

### P1-1 — `Clustering-Validation.ipynb`: شماره‌گذاری و سلسله‌مراتب
وضع فعلی: `### 1. Purity` ← `## Precision and Recall` ← `### 2. Maximum Matching` ←
`### 4. Pairwise Measures` (شمارهٔ ۳ وجود ندارد). همچنین `## Internal Validation Measures` با
`### Silhouette Coefficient` و بعد دو H2 ناهم‌سطح:
`## Python Example on Iris Dataset` و `## Interpretation of Results`.
**تغییر پیشنهادی:**

```
## External Measures
### 1. Purity
### 2. Precision, Recall and F-measure
### 3. Maximum Matching
### 4. Pairwise Measures (Jaccard, Rand, ARI)
## Internal Measures
### 5. Silhouette Coefficient
### 6. SSE / Inertia and the Gap Statistic
### 7. Choosing the Number of Clusters
## Choosing the Right Metric
```
عنوان‌های داخل متن که به شماره ارجاع می‌دهند (مثل «the examples on this page») بازبینی شوند.

### P1-2 — `Clustering-Validation.ipynb`: NMI تعریف نشده ولی چهار بار ارجاع داده شده
مکان‌های ارجاع: سلول‌های ۷، ۹، ۱۳، ۱۵، ۱۷ («purity, NMI and ARI will be close to 1») و در
جدول جمع‌بندی سلول ۴۸ ردیف ندارد.
**تغییر:** زیربخش کوتاه `#### Normalized Mutual Information (NMI)` بلافاصله بعد از ARI با:
تعریف بر پایهٔ آنتروپی مشترک، $I(C;T)/\sqrt{H(C)H(T)}$، بازهٔ $[0,1]$، و یک جمله که NMI
تصحیح‌شده برای شانس نیست (به همین دلیل AMI/ARI ترجیح داده می‌شوند). اگر D2 رد شد،
به‌جای تعریف، همهٔ ارجاع‌های NMI از متن حذف شوند.
همچنین یک ردیف `NMI | cluster | ❌ | ✅ | [0, 1]` به جدول سلول ۴۸ اضافه شود.

### P1-3 — `Clustering-Validation.ipynb`: معیارهای داخلی ناقص
مقدمه سه خانوادهٔ معیار را وعده می‌دهد؛ «Relative measures» هرگز نمی‌آید و از داخلی‌ها فقط
silhouette هست. برنامهٔ M2 صریحاً SSE و Gap statistic را می‌خواهد.
**تغییر:**
- زیربخش `#### SSE (Inertia)` با فرمول $\sum_i \|\mathbf{x}_i-\mu_{c(i)}\|^2$، اشاره به
  `KMeans.inertia_` و اینکه SSE با افزایش $k$ همیشه کم می‌شود ⇒ به‌تنهایی برای انتخاب $k$ بی‌فایده است؛
- زیربخش `#### Gap Statistic` در ۸ تا ۱۲ خط: ایده، $\mathrm{Gap}(k)=\mathbb{E}^*[\log W_k]-\log W_k$،
  و قاعدهٔ انتخاب کوچک‌ترین $k$ با Gap بزرگ‌تر از یک انحراف معیار؛
- یک پاراگراف کوتاه برای «Relative measures» (مقایسهٔ دو خوشه‌بندی با هم، بدون برچسب)،
  یا حذف آن از فهرست مقدمه.

### P1-4 — `Clustering-Validation.ipynb`: بخش «Further Reading» نامتناسب و بخشی غلط
سلول ۵۷ با ~۴٬۰۰۰ کاراکتر بزرگ‌ترین بلوک متنی نوتب‌وک است و دربارهٔ کاربردهای نامرتبط تطبیق
دوبخشی است (ازدواج پایدار، هم‌اتاقی، استاد راهنما، تطبیق تصویر، متقاضی شغل). در جدول پایانی
ادعاهای نادرست هست: «RANSAC» به‌عنوان الگوریتم تطبیق تصویر، و «هم‌اتاقی ← الگوریتم مجارستانی».
**تغییر (انتخاب یکی):**
- (الف، پیشنهادی) حذف سلول ۵۷ و جایگزینی با ۳ تا ۵ خط: تطبیق بیشینه یک مسئلهٔ تخصیص است و
  `scipy.optimize.linear_sum_assignment` آن را حل می‌کند؛ ارجاع به فصل ۱۷ زکی؛
- (ب) نگه‌داشتن حداکثر دو مثال، بدون جدول «Comparative Summary»، و با اصلاح دو ادعای بالا.

### P1-5 — `Clustering-Validation.ipynb`: تطبیق بیشینه چهار بار پیاده شده است
- سلول ۳۴: مثال میوه، دستی (خوب، نگه داشته شود)
- سلول ۳۵: `linear_sum_assignment` (خوب، نگه داشته شود)
- سلول‌های ۵۸–۶۲: سه سلول networkx با تابع `maximum_matching_bipartite_graph` که **دو بار**
  تعریف می‌شود؛ در سلول ۵۹ `bipartite.sets(G)` حساب و دور ریخته می‌شود؛ در سلول ۶۱ وزن یال‌ها
  نادیده گرفته می‌شود ولی برچسب «Weight» چاپ می‌شود؛ در سلول ۶۲ امضای تابع و منطق دوباره عوض می‌شود.
**تغییر:** فقط **یک** سلول networkx بماند (نسخهٔ وزن‌دار، با `weight='weight'`، بدون تعریف تکراری،
با توضیح اینکه `max_weight_matching` الزاماً تطبیق دوبخشی نمی‌سازد و برای گراف دوبخشی باید
پارتیشن‌ها را تضمین کرد). دو سلول دیگر حذف شوند.

### P1-6 — `Clustering-Validation.ipynb`: silhouette دوبار حساب می‌شود
سلول ۵۰ داخل «External/Pairwise example» مقدار silhouette را هم چاپ می‌کند (۰٫۵۵۱) و سلول ۵۳
همان محاسبه را تکرار می‌کند.
**تغییر:** محاسبه در بخش معیارهای داخلی بماند؛ در سلول ۵۰ فقط معیارهای بیرونی چاپ شوند.

### P1-7 — `Clustering-Validation.ipynb`: ابهام ترتیب آرگومان جدول توافقی
سلول ۶: `def compute_contingency_table(true_labels, cluster_labels)` که ماتریس `(r clusters × k classes)`
برمی‌گرداند؛ ولی سلول ۲۵: `purity_score` آن را با `compute_contingency_table(cluster_labels, true_labels)`
صدا می‌زند، و سلول ۵۰ با `compute_contingency_table(cluster_labels, y)`. چون هر دو آرگومان آرایه‌اند،
به‌ظاهر درست کار می‌کند و خواننده را گمراه می‌کند.
**تغییر:** یک امضای واحد، مثلاً `compute_contingency_table(cluster_labels, true_labels) -> (r, k)`
با docstring صریح `rows = clusters, cols = classes`، و اصلاح همهٔ فراخوانی‌ها
(سلول‌های ۸، ۱۰، ۱۲، ۱۴، ۱۶، ۱۸، ۲۰، ۲۳، ۲۵، ۵۰، ۶۱) تا با آن هم‌خوان باشند.

### P1-8 — `Clustering-Validation.ipynb`: دام «بهینه‌سازی معیار» نهفته مانده
سلول ۵۵ silhouette را برای $k=2..5$ حساب می‌کند و $k=2$ برنده می‌شود (۰٫۶۸۱) در حالی که $k$ واقعی
۳ است؛ خروجی `print(max(scores), scores.index(max(scores)))` هم `0.681… 0` چاپ می‌کند که مبهم است.
این دقیقاً «کلید» جلسهٔ M2 در برنامهٔ درس است.
**تغییر:** در `## Interpretation of Results` یک پاراگراف ۵ تا ۷ خطی: silhouette با $k=2$ برنده
می‌شود چون به جدایی خطی حساس است، در حالی که حقیقت ۳ است؛ نتیجه: هیچ معیار داخلی واحدی
«حقیقت» را کشف نمی‌کند و بهینه‌سازی یک معیار می‌تواند به انتخاب اشتباه بینجامد. خروجی چاپ
به `print({k: round(s, 3) for k, s in zip([2,3,4,5], scores)})` تغییر کند.

### P1-9 — `Clustering-Validation.ipynb`: قالب صفحه با سه صفحهٔ هم‌خانواده یکسان نیست
این صفحه نه بنر دارد، نه خط نویسنده/ترم، و با یک لینک به فصل ۱۷ زکی و `{cite}` **بالای** H1
شروع می‌شود.
**تغییر:** سلول اول به سبک دیگر صفحات: `![](img/banner.png)`، بعد H1، بعد
`**Mahmood Amintoosi, Fall 2026**` و خط گروه؛ و ارجاع به فصل ۱۷ به یک خط زیر عنوان منتقل شود.
همچنین یک بلوک «اهداف این جلسه» (۴ تا ۶ بولت) در ابتدا و یک «خلاصه» (۵ بولت) در پایان اضافه شود
تا با `High-Dimensional-Data` هم‌سبک شود.

### P1-10 — `Clustering-Part2.ipynb`: کم‌محتوا نسبت به جلسهٔ M1
برنامهٔ M1: ۳۰′ تعریف و SSE، ۳۰′ پیاده‌سازی از صفر، ۲۰′ لم اتحاد جفتی و همگرایی، ۱۰′ k-means++،
۱۰′ انتخاب k. نوتب‌وک لم‌ها و یک پیاده‌سازی دارد ولی سه مورد آخر غایب است.
**تغییر:**
- بخش `## Choosing the Initial Centroids: k-means++` (۱۰ تا ۱۵ خط): ضعف مقدار اولیهٔ تصادفی،
  نمونه‌گیری وزن‌دار $D^2$، و اینکه sklearn به‌صورت پیش‌فرض `init='k-means++'` است؛
- یک مثال کوچک از همگرایی به کمینهٔ محلی بد (مثلاً دو مرکزوار اولیهٔ نزدیک به هم) با شکل،
  اگر هزینه‌اش کم است؛ در غیر این صورت فقط توصیف یک‌خطی + اشاره به اینکه ترتیب داده اثر دارد؛
- پیچیدگی: $O(nkd)$ در هر تکرار، و اشاره به اینکه با `minibatch`/درخت‌ها می‌شود بهتر کرد
  (ارجاع به `Clustering-Part1` که `MiniBatchKMeans` را دارد)؛
- یک جمله در ابتدا که تقسیم کار با `Clustering-Part1` چیست: Part1 کاربرد/sklearn، Part2 نظریه و
  پیاده‌سازی از صفر.

### P1-11 — `Clustering-Part2.ipynb`: پیاده‌سازی سه‌بار و ایمپورت‌های مرده
- تابع `kmeans` (سلول ۲) هیچ‌جا صدا زده نمی‌شود؛
- کلاس `KMeansClustering` (سلول ۴) نسخهٔ دوم همان است؛
- `from sklearn.cluster import KMeans` در سلول ۴ ایمپورت می‌شود و استفاده نمی‌شود؛
- `from scipy.spatial import distance_matrix` در سلول ۶ (برای `lemma_2`) لازم است اما در همان
  سلول ایمپورت شده — بررسی شود که با حذف تابع تکراری نشکند.
**تغییر:** یکی بماند. پیشنهاد: تابع تابعی به‌عنوان «naive first version» با یک جمله توضیح نگه داشته
شود و کلاس حذف شود، **یا** برعکس؛ در هر حال ایمپورت بی‌استفاده حذف و یک `SEED = 42` بالای نوتب‌وک.

### P1-12 — `Clustering-Part2.ipynb`: پیوند نامرتبط در فهرست مواد تکمیلی
در سلول ۱۰، مقالهٔ فارسی «Challenges and Requirements of Persian Language Support in Citation Software»
زیر بولت «Weiszfeld's algorithm» تودرتو شده و به k-means ربطی ندارد.
**تغییر:** حذف آن خط، یا انتقال به انتهای فهرست بدون تودرتو.

### P1-13 — `High-Dimensional-Data.ipynb`: $d$ و $p$ هم‌زمان برای بُعد
مقدمه «$n \times d$ matrix» می‌نویسد، چند خط بعد «$p$-dimensional unit hypercube» و $e_p(r)$.
**تغییر (D1):** $d$ در همه‌جا؛ $e_d(r) = r^{1/d}$؛ زیرنویس شکل ESL هم که $p$ دارد با یک جمله
«(the figure labels the dimension $p$)» توضیح داده شود.

### P1-14 — `High-Dimensional-Data.ipynb`: گوی و کره یکی گرفته شده‌اند
$S_d(r)$ این‌طور تعریف می‌شود: «all points exactly at distance $r$ from the center» (یعنی سطح)،
ولی بعد `Vol(S_d(r))` برای حجم گوی به‌کار می‌رود.
**تغییر:** $B_d(r)$ برای گوی (حجم $K_d r^d$) و $S^{d-1}(r)$ برای سطح؛ همهٔ ارجاع‌های
`\text{Vol}(S_d(...))`، `\text{Vol}(S_d(r,\epsilon))` و نسبت پوسته اصلاح شوند.

### P1-15 — `High-Dimensional-Data.ipynb`: قید «تقریباً ۱۰۰٪» بی‌قید
جملهٔ «almost 100% of the volume … is concentrated in a thin shell» فقط برای $\epsilon/r$ ثابت
درست است.
**تغییر:** «for a shell of fixed relative thickness $\epsilon/r$».

### P1-16 — `High-Dimensional-Data.ipynb`: فرمول $d(p,N)$ بی‌مبنا
سلول ۶ فرمول $d(d,N) = (1-(1/2)^{1/N})^{1/d}$ را بدون هیچ استنتاج یا ارجاع می‌آورد، در حالی که
بقیهٔ نوتب‌وک همه‌چیز را استدلال می‌کند.
**تغییر:** دو خط استدلال: احتمال اینکه هیچ‌یک از $N$ نقطه در گوی شعاع $t$ نباشد برابر
$(1-t^d)^N$ است؛ برابر گذاشتن با $1/2$ و حل برای $t$ همان فرمول را می‌دهد. ارجاع به ESL §2.5.

### P1-17 — `High-Dimensional-Data.ipynb`: بخش تولید نقطه، روایت را قطع می‌کند و تکرار دارد
سلول‌های ۹–۱۹ (~۹٬۵ هزار کاراکتر متن+کد) دربارهٔ «نمونه‌گیری یکنواخت» است، نه یکی از پنج پدیدهٔ
مقدمه؛ و نمونه‌گیری معکوس دوبار توضیح داده می‌شود (سلول ۱۶ فشرده، سلول ۱۸ مفصل).
**تغییر:** عنوان آن به `## Aside: Sampling Points Uniformly in a Ball` تغییر کند (تا خواننده بداند
انحراف است) و سلول‌های ۱۶ و ۱۷ با سلول ۱۸ یکی شوند (توضیح مفصل بماند، تکرار حذف شود). اگر حجم
باز هم زیاد بود، مثال «نقض ساده‌لوحانه» کوتاه شود؛ محتوای ریاضی (CDF، جدول $d=2$) حفظ شود.

### P1-18 — `High-Dimensional-Data.ipynb`: پدیدهٔ غایب «تمرکز فاصله‌ها»
مقدمه پنج پدیده وعده می‌دهد و مهم‌ترین پدیده برای kNN — نزدیک‌شدن نسبت فاصلهٔ نزدیک‌ترین به
دورترین به ۱ — نیامده است؛ همین پدیده موضوع صفحهٔ چهارم است.
**تغییر:** زیربخش `## Distance Concentration` بعد از «The Problem of Local Neighborhoods» با:
تعریف نسبت $d_{\min}/d_{\max}$، آزمایش ۶ خطی روی دادهٔ یکنواخت برای $d=2,10,100,1000$،
یک شکل، و جملهٔ نتیجه: در بُعد بالا «نزدیک‌ترین» همسایه تقریباً به همان دوری «دورترین» است،
پس رتبه‌بندی همسایه‌ها بی‌معنا می‌شود.

### P1-19 — `High-Dimensional-Data.ipynb`: برچسب‌های کهنه و مقایسهٔ نظری انجام‌نشده
در سلول ۱۰ `num_points = 2000` است ولی تیتر نمودار «Ratio of 10000 / N for d = 2 to 10» و
`ylabel='Ratio of 10000 to Total Generated Points'` است.
**تغییر:** برچسب‌ها به ۲۰۰۰ اصلاح شوند و در همان شکل منحنی نظری $K_d/2^d$ هم رسم شود
(نسبت حجم گوی واحد به مکعب $[-1,1]^d$)، با legend «measured» و «theory». این دقیقاً جای
مقایسهٔ عدد و فرمول است که الان از دست رفته.

### P1-20 — `High-Dimensional-Data.ipynb`: سلسله‌مراتب تیترها
سلول ۲۰ با `---` و متن شروع می‌شود و `### Random Vectors` بدون H2 والد می‌آید؛
سلول ۲۲ `## Why Gaussian Points Do *Not* Cluster Near the Mean (Unlike 1D)` را در سطح H2 می‌آورد
در حالی که زیربخش `## Gaussian Distribution in High Dimensions` است.
**تغییر:** سلول ۲۰ ← `## Nearly Orthogonal Directions` و داخلش `### The Diagonal of the Hypercube`
و `### Random Vectors`؛ سلول ۲۲ ← `### Why Gaussian Points Do Not Cluster Near the Mean`.

### P1-21 — `High-Dimensional-Data.ipynb`: سلول‌های وابسته به ترتیب و بدون seed
`dimensions = range(1,30)` در سلول ۲ تعریف و در سلول ۸ استفاده می‌شود؛ `hypersphere_volume` هم
همین‌طور؛ و هیچ `seed` وجود ندارد، پس شکل‌ها هر اجرا تفاوت می‌کنند.
**تغییر:** در سلول اول `SEED = 42` و `np.random.default_rng(SEED)`؛ `dimensions` در همان سلولی
تعریف شود که استفاده می‌شود (یا در یک سلول تنظیمات بالای نوتب‌وک)؛ `hypersphere_volume` به
همان سلول تنظیمات منتقل شود.

### P1-22 — `High-Dimensional-Data.ipynb`: وعدهٔ «پل به JL»
صفحه با «This section therefore serves as the bridge … to the Johnson–Lindenstrauss Lemma» تمام
می‌شود ولی صفحه‌ای برای JL در درس نیست.
**تغییر (D4):** جمله به «the properties we need when we come back to random projections» تغییر کند
و یک جملهٔ واقع‌گرا دربارهٔ ادامهٔ درس. صفحهٔ جدید ساخته نشود.

### P1-23 — `High-Dimensional-Data-and-kNN.ipynb`: ادغام یا بازنویسی
این صفحه ۲۱۴ کاراکتر متن دارد و هیچ توضیح، تفسیر، مرجع یا جمع‌بندی ندارد؛ آزمایشش هم پدیدهٔ
موردنظر را نشان نمی‌دهد: با `n_informative=2` و `n_redundant=0` ویژگی‌های اضافی فقط نویزند و
منحنی دقت صاف می‌ماند.
**تغییر (D3، پیشنهادی):** ادغام در `High-Dimensional-Data` به‌عنوان
`## k-NN under the Curse of Dimensionality` با آزمایش درست:
- منحنی دقت kNN برحسب $d$ با **ثابت‌نگه‌داشتن $n$** و افزایش $d$ (که افت نشان می‌دهد)،
- همان منحنی وقتی $n$ با $d$ رشد می‌کند (که افت را جبران می‌کند)،
- تکرار روی چند seed و رسم میانگین ± انحراف.
سپس فایل حذف و ورودی آن از TOC برداشته شود.
اگر استاد ادغام را نخواست: صفحه باید با اهداف، سه بخش، یک آزمایش درست و جمع‌بندی بازنویسی شود.

---

## ۴. P2 — صیقل

- **P2-1 (عنوان‌ها):** TOC عنوان صفحه را از H1 می‌گیرد، پس `Clustering-Part2` به‌عنوان عنوان در
  فهرست دیده می‌شود. پیشنهاد: H1های `k-Means: Objective, Lemmas and Lloyd's Algorithm`،
  `Clustering Validation`، `High-Dimensional Data and the Curse of Dimensionality`،
  و برای صفحهٔ kNN در صورت ماندن `k-NN in High-Dimensional Spaces`. سبک Title Case یکدست شود.
- **P2-2 (زیرنویس شکل‌ها):** در `High-Dimensional-Data` زیرنویس‌ها پاراگراف معمولی‌اند
  («Figure 6.4. …»). به یک الگوی واحد تبدیل شوند: خط ایتالیک کوچک زیر تصویر، با ذکر منبع
  (Zaki ch. 6 / ESL fig. 2.6).
- **P2-3 (ارجاع‌ها):** در `High-Dimensional-Data` فهرست دستی References هست؛ برای
  `hastie2009elements` و `zaki2020data` که در `notebooks/references.bib` موجودند از `{cite}`
  استفاده شود و فهرست دستی به دو مرجعی که کلید ندارند محدود شود.
- **P2-4 (کیفیت شکل):** `plt.rcParams['figure.dpi'] = 600` در `Clustering-Validation` سلول ۰
  به ۱۵۰ کاهش یابد (یا فقط برای شکل‌های نهایی ست شود)؛ سنگینی صفحه محسوس است.
- **P2-5 (املای برچسب‌ها):** `dimention` (دو جا)، `voumle_shell` و
  `voumle_shell/volume_hypersphere` (هدر و legend)، `Generate points form Gaussian distribution`،
  `implememntation` در `Clustering-Part2` سلول ۳، `Silhouetter Score` در سلول ۵۳.
- **P2-6 (وابستگی‌ها):** `yellowbrick`، `networkx`، `seaborn` بی‌مستند استفاده شده‌اند و
  `# !pip install yellowbrick` کامنت است. یک خط در ابتدای هر نوتب‌وک که به کدام بسته‌ها نیاز دارد،
  یا اشاره به صفحهٔ نصب درس.
- **P2-7 (اهداف و جمع‌بندی):** هر چهار صفحه یک بلوک «Learning objectives» در ابتدا و یک
  «Summary / Checklist» در پایان داشته باشند (الگوی موجود در `High-Dimensional-Data` §Summary).
- **P2-8 (تمرین):** در `Clustering-Validation` یکی از هفت مثال (مثلاً مثال E، جدول یکنواخت)
  به سلول ناتمام تبدیل شود: جدول داده شود و محاسبهٔ purity/ARI به دانشجو سپرده شود.

---

## ۵. قرارداد نمادها (پس از اعمال D1)

| کمیت | نماد | یادداشت |
|------|------|---------|
| تعداد نقاط | $n$ | |
| بُعد | $d$ | در شکل‌های ESL با $p$ برچسب خورده؛ در متن $d$ |
| تعداد خوشه‌ها | $k$ | همان `n_clusters` در sklearn |
| تعداد کلاس‌های حقیقی | $r$ | در زکی $k$ است؛ در `Clustering-Validation` تغییر کند |
| بردار | $\mathbf{x}_i$ | نه $\textbf{x}$ و نه $x$ |
| نرم | $\|\cdot\|$ | نه $|\cdot|$ |
| گوی / کره | $B_d(r)$ / $S^{d-1}(r)$ | حجم گوی $K_d r^d$ |
| جدول توافقی | $n_{ij}$ | $i$ = خوشه، $j$ = کلاس |

اگر این قرارداد اعمال شد، در `Clustering-Validation` یک بلوک `#### Notation` کوتاه اضافه شود:
«Zaki writes $r$ for clusters and $k$ for classes; this page uses the course convention
$k$ clusters, $r$ classes.»

---

## ۶. چک‌لیست تأیید (قبل از تحویل)

1. **بیلد سایت بدون هشدار برای این صفحه‌ها:**
   ```bash
   npx --yes mystmd@latest build --site
   ```
   انتظار: صفر هشدار `No target for internal reference` / `unknown directive` برای این چهار صفحه.
   (هشدارهای موجود `MLP-and-Backpropagation.ipynb` مربوط به این کار نیستند.)
2. **حساب جدول‌ها:** با یک اسکریپت کوچک، جمع سطر/ستون و purity/ match/ F دو جدول سلول‌های ۳۸ و ۳۹
   بازمحاسبه و با متن جدول مقایسه شود. هر عددی که هم‌خوان نبود، اصلاح شود.
3. **اجرای مجدد نوتب‌وک‌های تغییرکرده** و بررسی اینکه خروجی ذخیره‌شده بدون هشدار و بدون `nan` است.
4. **جست‌وجوی باقی‌مانده‌ها:**
   ```bash
   grep -n "%%HTML\|require.js" notebooks/Clustering-Part2.ipynb notebooks/High-Dimensional-Data.ipynb notebooks/High-Dimensional-Data-and-kNN.ipynb
   grep -n "implememntation\|dimention\|voumle_shell\|form Gaussian" notebooks/*.ipynb
   grep -n "0.667\|n = 100" notebooks/Clustering-Validation.ipynb
   ```
   هر سه باید خالی برگردند.
5. **یکدستی نماد:** `grep -n "S_d(" notebooks/High-Dimensional-Data.ipynb` نباید برای حجم گوی
   باقی بماند؛ `grep -n "\\max_{j=1}^k" notebooks/Clustering-Validation.ipynb` نباید برای argmax بماند.
6. **TOC:** اگر D3 اعمال شد، ورودی `High-Dimensional-Data-and-kNN.ipynb` از `myst.yml` حذف و
   صفحه از خروجی بیلد غایب باشد.

## ۷. خارج از دامنه

- `Clustering-Part1.ipynb` (sklearn/EM/ارقام/فشرده‌سازی رنگ): فقط در یک جملهٔ تقسیم کار به آن
  ارجاع داده می‌شود؛ محتوایش بازبینی نشود.
- `css/rtl.css`، فایل‌های فارسی `course-fall-2026/`، `notebooks/references.bib`.
- ساخت صفحهٔ جدید برای لم جانسون–لیندن‌شتراوس (تنها با درخواست صریح استاد).
- تغییر متن برنامهٔ درس `course-fall-2026/course-plan-fall-2026.md`؛ اگر محتوای جلسه عوض شد،
  فقط پیشنهاد آن در گزارش پایانی ذکر شود.

## ۸. گزارش پایانی مورد انتظار

برای هر شناسه (`P0-1` … `P2-8`): وضعیت (انجام شد / انجام نشد و چرا)، فایل و سلول، و اگر عددی
عوض شد مقدار قبل و بعد. مواردی که به تصمیم استاد نیاز دارد (D1…D5) جدا فهرست شوند.
