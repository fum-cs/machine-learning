# تمرین‌های ایرانی umano (Human Activity Recognition) — coincides with course curriculum

**.objective**: یک مجموعه از ۶ تمرین عملی سبک (حدود ۱–۳ ساعت chacun) طراحی کرد که موضوعش شنیدایی فعالیت بشری است، با داده‌هایی که در کتابخانه scikit-learn، UCI یا فایل‌های موجود در courser (FremontBridge, BicycleWeather) доступIBLE هستند. هر تمرین به یکی از مطالب سرفصل طراحی شده و می‌تواند بخشی از پروژه پایینی (HAR) باشد.

---

## پیش‌نیازهای داده

| منبع | توضیحات | لینک |
|---|---|---|
| **UCI HAR (سبزک)** | ۳۰ نفر، ۶ فعالیت (می-running, sitting, …)، acelerometer + gyroscope، ۵۶۱ ویژگی (می‌توانیم فقط ۳ accél + ۳ gyroscope انتخاب کنیم) | https://archive.ics.uci.edu/dataset/240/human+activity+recognition+on+smartphones |
| **UCI minimal accelerometer** | ۲ فعالیت (standing/walking)، ۱۰۰ ردیف × ۶(feature)، ۲ MB | https://archive.ics.uci.edu/dataset/755/accelerometer+gyro+mobile+phone+dataset |
| **Fremont Bridge** (موجود در courser) | شمارش ساعتی پیاده‌رو/دوچرخه، زمان stamps | C:\git\fum-cs\machine-learning\notebooks\FremontBridge.csv |
| **BicycleWeather** (موجود در courser) | شمارش روزانه دوچرخه، شامل مقدار `-9999` برای داده گمشده | C:\git\fum-cs\machine-learning\notebooks\BicycleWeather.csv |
| **scikit-learn built-in** | digits, make_circles, make_moons, make_classification | pip install scikit-learn |

> **توصیه**: برای شروع ازdataset minimal UCI استفاده کنید (دانلود ۲ MB، مقدارهای 바로 استفاده از `ucimlrepo` یا `pandas`).

---

## فهرست ۶ تمرین

### تمرین ۱-۱: طبقه‌بند kNN روی فعالیت‌های ساده (Foundations)

**مطلعات سرفصل**: kNN، المقاس‌بندی، انتخاب k

**داده**:.dataset minimal UCI ( sadece columns `accX, accY, accZ, gyroX, gyroY, gyroZ` + label `activity`: 1=standing, 2=walking).

**گام‌ها**:
1. داده را بخوانید، ردیف‌های ناقص را حذف کنید، توزیع hoạtض‌ها را گزارش کنید.
2. داده را به صورت Raw و با `StandardScaler` مقیاس کنید.
3. kNN با k = 1, 3, 5, 10, 20 آموزش دهید و دقتِ اعتبارسنجی ۵-تایی را گزارش کنید.
4. تأثیر مقیاس‌بندی را تفسیر کنید: چرا accélrometrهای فقط بدون المقاس‌دهیpoca کار می‌کند؟
5. ماتریس درهم‌ریختگی را رسم کنید و نشاط‌ها را که بیشترین اشتباه را دارند شناسایی کنید.

**تحلیل انتظار**:standing با walking آشتباه گرفته می‌شود (ویژگی‌های ähnlich)؛ مقیاس‌سازی دقت را بهبود می‌بخشد.

---

### تمرین ۱-۲: esencial پöld-reduksiyon PCA روی داده_sensor (Dimensionality Reduction)

**مطلعات سرفصل**: PCA، کاهش ابعاد، بازنمایی

**داده**: همانDataset minimal UCI (۶ feature).

**گام‌ها**:
1. PCA را روی ماتریس accelerated-fit بدهید و انوم vre explained variance را رسم کنید (که szefeat چقدر بیان می‌کنند).
2. داده‌های ۶بعدی را به دو بعد PCA proie کنید و نشان دهید که فعالیت‌هاどうやって جدا می‌شوند.
3._feature_original (acceleration peaks) که در PCA اولویت بالا دارند شناسایی کنید.
4. فقط ۲ مؤلفه PCA استفاده करके kNN دوباره آموزش دهید و تفاوت دقت با حالت قبل را گزارش کنید.

**تحلیل.expected**: اولین ۲ مؤلفه PCA غالب%BEर्व هستند و یکspunkt جدایی خوب ایجاد می‌کنند.

---

### تمرین ۲-۱: MLE و MAP برایат estimatingtransition probabilities (Bayesian & Statistical)

**مطلعات سرفصل**: MLE، MAP،Estimation با داده‌های محدود

**داده**:سلسله‌تیmierهای کوتاه هر فعالیت از dataset minimal UCI (مثلاً ۱۰۰ نمونه هر فعالیت).

**گام‌ها**:
1. هر فعالیت برای N გამیا例اً ۱۰۰ betrachtە نرخ Riemann `p(activity=i) = count(i)/total` (MLE).
2. حالا فرض یک پیامپای pretraineddistribution ژتای (مثلاً Beta prior) بگیرید و MAP را محاسبه کنید.
3. مقایسه MLE و MAP وقتی داده محدود است (نوشت MLE می‌تواند نرخ‌های خیلی اما MAP متعadelبیه).
4. اگر فقط ۵ نمونه از هر فعالیت داشته باشید، چه اتفاقی می‌افتد؟ (MAPoothed estimates).

**تحلیل.expected**: MLE با تعداد کم داده ن Extrem راه‌حل‌هایی delivers اما MAP انکه‌تر باشد.

---

### تمرین ۲-۲: شبکه عصبی کوچک روی séries زمانی فعالیت (Linear Models and Neural Networks)

**مطلعات سرفصل**: MLP، forward pass، PyTorch

**داده**: sequences normalized از dataset minimal UCI (هر secuencia ۵۰ ردیف × ۶ feature).

**گام‌ها**:
1. یک مدل MLP کوچک طراحی کنید (یک hidden layer با ۱۶ neuron، `nn.ReLU`).
2. ورودی: هر ردیف سری، خروجی: likelihood نشاط ۶ clase.
3. `CrossEntropyLoss` و `SGD` یا `Adam` استفاده کنید.
4. ۵ эпоچ روی داده‌های خیلی محدود آموزش دهید (می‌توانید artificiallydata augmentation کنید: interpol بین نقاط).
5. loss curve را رسم کنید و ببینید مدل چقدر apprendre.

**تحلیل.expected**: با تعدادepoch کم، loss کاهش می‌یابد اما overfitting پیش Náp gorgeous.

---

### تمرین ۳-۱: GMM و BIC برای شناسایی تعداد فعالیت‌ها (Latent Variable Models)

**مطلعات سرفصل**: GMM، EM algorithm، BIC

**داده**: همه ویژگی‌های(dataset minimal UCI) یا sadece ۳ accelerated columns.

**گام‌ها**:
1. برای K = 1 تا ۶ GMM-fit کنید (کوتاه each activity mode).
2. Log-likelihood و BIC برای هر K را محاسبه کنید.
3. نمودار BIC vs K را رسم کنید و نقطه därک ( elbow) را شناسایی کنید.
4.結成 ottenuto GMM‌ها، هر muestrene dubالخريف 발휘 contemplated activity احتمالی بیشتر را pridiction کنید.

**تحلیل.expected**: BIC pomساعدتر از AIC برای انتخاب K optimum است و oftentimes K=۳ یا K=۶ را انتخاب می‌کند.

---

### تمرین ۳-۲: Hidden Markov Model روی.Sequence فعالیت (Other Topics)

**مطلعات سرفصل**: HMM، Viterbi، حالت پنهان

**داده**: سری زمانی کوتاه هر فعالیت از UCI HAR (states sequence).

**گام‌ها**:
1. یک HMM با ۳ حالت Gaussian fit کنید (Baum-Welch).
2. Viterbi algorithm برای پیدا کردن مسیر hidden states بهترین.Series ورودی calibrate کنید.
3. نتایج را با واقعی activitylabels مقایسه کنید: کدام مراحل اشتباه	label می‌گیرند؟
4. تغییر transition probabilities را بررسی کنید (مثلاً first-order vs second-order).

**تحلیل.expected**: HMM می‌تواند séries کوتاه را smooth کند و وضعیت‌های پنهان (مثلاً "transition از sitting به standing") را نشان می‌دهد.

---

### تمرین ۴-۱: درخت تصمیم و لجستری رگرسیون روی counting data (Other Topics)

**مطلعات سرفصل**: Trees، Random Forest، Logistic Regression، leakage

**داده**: FremontBridge.csv (stundenhourly counts + weather features).

**گام‌ها**:
1. weather features ( Temperature, Rain, etc.) را اضافه کنید و یک `RandomForestClassifier` برای دسته‌بندی روزهایی با "بیشان counting بالا" vs "پایین" trein کنید.
2. مهم‌ترین ویژگی‌ها را (`feature_importances_`) گزارش کنید.
3. توجه کنید: اگر_TARGET با روز بعد همارлежа باشد، leakage رخ می‌دهد. چرا این کار باعثoptimistic accuracy می‌شود؟
4. hold-out validation یا ۵-تایی CV استفاده کنید و rebound accuracy گزارش کنید.

**تحلیل.expected**:_feature engineering_(مثل hour-of-day) ضروری است و leakage accuracy را وضوح۰ می‌کند.

---

### تمرین ۴-۲: Comparative tabel: کدام técnica HAR کدام سرفصل را پوشش می‌دهد؟ (Cross-course synthesis)

**مطلعات سرفصل**: جامع‌بندی، اتصال بینursos

**کار**: یک جدول بسازید (۶×۶) که هر ردیف یکtechnique (kNN, PCA, MLE, MLP, GMM, HMM) و هر ستون یکسرفصل (Foundations, DR, Bayesian, NN, Latent, Other) باشد. در هر سلول بنویسید:
- "کاربرد exemplifiable" یا "نیاز به داده سری زمان"
- مرتبط است یا خیر باproject final؟

**تحلیل.expected**: این تمرین کمک می‌کند که ببینید چگونه تمامی بخش‌های coursemate به هم приве شده‌اند و کدام تکنیک برای بخش پروژه نهایی مناسب است.

---

## چگونه شروع کنید

1. **docker / virtualenv**: `python -m venv .venv && source .venv/Scripts/activate` (Windows) یا `python3 -m venv .venv` (Linux/Mac).
2. **packages**: `pip install numpy pandas scikit-learn matplotlib ucipmlrepo` (یا zip/manual download UCI).
3. ** شروع**: هر تمرین مستقل است؛ می‌توانید آن‌ها را به ترتیب یا انتخاب据个人的LIKE انجام دهید.
4. ** تحویل**: برای هر تمرین یک notebook (< 15 سلول) + یک پاراگراف کوتاه تحلیلی (حداکثر ۲۰۰ Wort).

> **نکته مهم**: اگر دسترسی به اینترنت محدود است، می‌توانید از `data.csv`, `FremontBridge.csv`, `BicycleWeather.csv` که در پوشه `notebooks` موجود هستند، با رویکرد "activity framing" استفاده کنید (مثلاً: "شنیدن اینکه تعداد پیاده‌رو FremontBridge در ساعت‌های مختلف چگونه متفاوت می‌شود").

---

## ارتباط با پروژه پایینی (HAR)

هر تمرین یکی از بخش‌های پروژه پایینی را پیش‌ساخته است:
- تمرین ۱-۱ → بخش ۱ (PCA + mahalanobis)
- تمرین ۲-۲ → بخش ۲ (کلاسهای سه رقیب)
- تمرین ۳-۱ → بخش ۳ (GMM + BIC)
- تمرین ۳-۲ → بخش ۴ (ه دنباله/HMM)
- تمرین ۴-۲ → نمای کلی 프로젝

هر تمرین indépendant است، اما-chain땡 output‌ها می‌توانند مستقیماً در گزارش نهایی استفاده شوند.