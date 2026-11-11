# مراجعِ ساخته‌شده — برای بازبینی مدرس

پس از رفع هشدارهای `Could not link citation` در ۱۶ مدخل `notebooks/references.bib`، آن‌ها دو دسته‌اند:

## الف) ۱۰ مدخل که از `C:\git\fum-cs\data-mining\notebooks\references.bib` برداشته شدند

`ESLII_print12` · `Duda2000` · `theodoridis2020machine` · `Donoho2006CS` · `gueddari2020PySAP` ·
`saleh2019theory` · `saleh2022rank` · `Rashidabadi2016image` · `Asaran2016SR` · `Bakhshali2018Sparse`

این‌ها **متن تألیفی من نیستند**؛ فقط فیلدهای اضافی (مثل `hal_id`/`pdf` در PySAP) حذف و قالبشان
یکدست شد. دو استثنا که ارزش بازبینی دارند:

- **`ESLII_print12`** همان کتاب `hastie2009elements` است (Elements of Statistical Learning، ویرایش ۲،
  ۲۰۰۹) که از قبل در همین فایل بود. چون `Naive-Bayes.ipynb` دقیقاً این کلید را صدا می‌زند، مدخل
  جداگانه با یک کامنت بالای آن نگه داشته شد. اگر ترجیح می‌دهید تک‌نسخه باشد، کافی است ارجاع در
  نوتبوک را به `hastie2009elements` تغییر دهید و این مدخل را حذف کنید.
- **`gueddari2020PySAP`** در فایل مبدأ ۸ نویسنده با فیلدهای HAL داشت؛ من فقط `booktitle` را
  ساده کردم.

## ب) ۶ مدخلی که خودم نوشتم و باید تأیید شوند

این‌ها در هیچ‌کدام از مخزن‌های `fum-cs` پیدا نشدند و از دانش عمومی/جست‌وجوی وب ساخته شدند.

| کلید | چه‌چیزی تأیید شد | چه‌چیزی تأیید نشد |
|---|---|---|
| `vanderplas2016python` | نویسنده، ناشر (O'Reilly)، سال ۲۰۱۶، وبگاه کتاب | شمارهٔ ISBN |
| `bemis1996properties` | نشریه، جلد ۳۹، شمارهٔ ۱۵، صفحات ۲۸۸۷–۲۸۹۳، DOI | — |
| `wu2018moleculenet` | نشریه (Chemical Science)، جلد ۹، شمارهٔ ۲، صفحات ۵۱۳–۵۳۰، DOI، فهرست نویسندگان | — |
| `yang2019analyzing` | نشریه (JCIM)، جلد ۵۹، شمارهٔ ۸، صفحات ۳۳۷۰–۳۳۸۸، DOI، فهرست بلند نویسندگان | — |
| `guo2024scaffold` | عنوان، سه نویسنده، ICANN 2024، صفحات ۵۸–۷۲، DOI انتشارات Springer، arXiv:2406.00873 | شمارهٔ LNCS |
| `Casalicchio2024Evaluation` | نویسندگان (Casalicchio و Burk)، عنوان فصل، صفحات ۵۳–۸۲، ناشر Chapman & Hall/CRC، سال ۲۰۲۴ | **نام ویراستاران** (از حافظه نوشتم)، شمارهٔ فصل |

### متن مدخل‌ها همان‌طور که در `notebooks/references.bib` است

```bibtex
@book{vanderplas2016python,
  title     = {Python Data Science Handbook: Essential Tools for Working with Data},
  author    = {VanderPlas, Jake},
  publisher = {O'Reilly Media},
  year      = {2016},
  isbn      = {9781491912058},
  url       = {https://jakevdp.github.io/PythonDataScienceHandbook/}
}

@article{bemis1996properties,
  title   = {The Properties of Known Drugs. 1. Molecular Frameworks},
  author  = {Bemis, Guy W. and Murcko, Mark A.},
  journal = {Journal of Medicinal Chemistry},
  volume  = {39},
  number  = {15},
  pages   = {2887--2893},
  year    = {1996},
  doi     = {10.1021/jm9602928}
}

@article{wu2018moleculenet,
  title   = {MoleculeNet: a benchmark for molecular machine learning},
  author  = {Wu, Zhenqin and Ramsundar, Bharath and Feinberg, Evan N. and Gomes, Joseph and
             Geniesse, Caleb and Pappu, Aneesh S. and Leswing, Karl and Pande, Vijay},
  journal = {Chemical Science},
  volume  = {9},
  number  = {2},
  pages   = {513--530},
  year    = {2018},
  doi     = {10.1039/C7SC02664A}
}

@article{yang2019analyzing,
  title   = {Analyzing Learned Molecular Representations for Property Prediction},
  author  = {Yang, Kevin and Swanson, Kyle and Jin, Wengong and Coley, Connor and Eiden, Philipp and
             Gao, Hua and Guzman-Perez, Angel and Hopper, Timothy and Kelley, Brian and Mathea, Miriam and
             Palmer, Andrew and Settels, Volker and Jaakkola, Tommi and Jensen, Klavs and Barzilay, Regina},
  journal = {Journal of Chemical Information and Modeling},
  volume  = {59},
  number  = {8},
  pages   = {3370--3388},
  year    = {2019},
  doi     = {10.1021/acs.jcim.9b00237}
}

@inproceedings{guo2024scaffold,
  title     = {Scaffold Splits Overestimate Virtual Screening Performance},
  author    = {Guo, Qianrong and Hernandez-Hernandez, Saiveth and Ballester, Pedro J.},
  booktitle = {Artificial Neural Networks and Machine Learning -- ICANN 2024},
  series    = {Lecture Notes in Computer Science},
  pages     = {58--72},
  year      = {2024},
  publisher = {Springer, Cham},
  doi       = {10.1007/978-3-031-72359-9_5}
}

@incollection{Casalicchio2024Evaluation,
  title     = {Evaluation and Benchmarking Applied Machine Learning Using mlr3 in R},
  author    = {Casalicchio, Giuseppe and Burk, Lukas},
  booktitle = {Applied Machine Learning Using mlr3 in R},
  editor    = {Bischl, Bernd and Sonabend, Raphael and Kotthoff, Lars and Lang, Michel},
  pages     = {53--82},
  year      = {2024},
  publisher = {Chapman and Hall/CRC}
}
```

### جاهایی که ارجاع می‌شوند

- `vanderplas2016python` و `ESLII_print12` ← `notebooks/Naive-Bayes.ipynb`
- `Duda2000` ← `Mahalanobis-Distance.ipynb` (سه‌بار) و `Bayesian-Decision-Theory.ipynb`
- `guo2024scaffold`، `bemis1996properties`، `wu2018moleculenet`، `yang2019analyzing` و
  `Casalicchio2024Evaluation` ← `notebooks/Model-Evaluation.ipynb`
- بقیهٔ مدخل‌های واردشده ← `Linear-Models-Advanced-Topics.ipynb`

### بررسی سریع

```bash
# صفر هشدار ارجاع در بیلد سایت
npx mystmd build --site 2>&1 | grep -ci citation     # باید 0 باشد
```

اگر مدخلی را نپسندیدید یا فیلدی غلط بود، فقط همان بلوک را در `notebooks/references.bib`
اصلاح کنید؛ هیچ‌جای دیگری به این فیلدها وابسته نیست.
