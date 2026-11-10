# دستور کار راست‌به‌چپ در MyST v2 (Jupyter Book 2)

این نسخهٔ قابل‌انتقالِ کاری است که در همین مخزن انجام شد. برای عامل هوشمند دیگری نوشته شده که
می‌خواهد همان کار را در مخزن دیگری تکرار کند.

## چرا روش قدیمی کار نمی‌کند

در Jupyter Book **v1** دو راه بود:

1. فرادادهٔ صفحه: `html_meta: {dir: rtl, lang: fa}`
2. پیچیدن متن در HTML: `<div dir="rtl" markdown="1"> ... </div>`

در **v2** (mystmd) هیچ‌کدام کار نمی‌کند:

- `html_meta` دیگر کلید شناخته‌شده‌ای نیست؛ myst هشدار می‌دهد
  `'frontmatter' extra key ignored: html_meta` و آن را دور می‌ریزد.
- markdown داخل HTML خام **پردازش نمی‌شود**؛ نتیجه یک `div` توخالی است و متن فارسی همان‌جا
  دست‌نخورده و چپ‌چین می‌ماند (یا کل بلوک به‌صورت متن خام نمایش داده می‌شود).

mystmd هم (تا نسخه ۱٫۱۲) گزینهٔ `dir`/`lang` در frontmatter یا `site.options` ندارد. پس راه درست،
ترکیب دو چیز است: **یک کلاس در بدنهٔ صفحه** + **یک فایل CSS سراسری**.

## گام ۰ — پیش‌نیاز

مطمئن شوید پروژه MyST v2 است: فایل `myst.yml` با `version: 1` و `site:` وجود دارد و بیلد با
`myst build --site` یا `jupyter-book build --html` (بستهٔ npm) انجام می‌شود. اگر پروژه هنوز
`_config.yml`/`_toc.yml` دارد، اول مهاجرت لازم است و این دستور کار صدق نمی‌کند.

## گام ۱ — فایل CSS

`css/rtl.css` را بسازید (مسیر دلخواه است، ولی همین ساختار ساده بماند):

```css
/* فقط عناصر دارای کلاس rtl را جهت می‌دهد؛ صفحه‌های انگلیسی دست‌نخورده می‌مانند. */
.rtl {
  direction: rtl;
  text-align: right;
}

/* پشتهٔ قلم فارسی. قاعده را روی فرزندها هم تکرار کنید، وگرنه قلم پایهٔ تم
   با ترتیب بارگذاری برنده می‌شود. Tahoma روی ویندوز فارسی را خوب می‌دهد. */
.rtl,
.rtl p, .rtl li, .rtl td, .rtl th,
.rtl h2, .rtl h3, .rtl h4,
.rtl figcaption, .rtl blockquote, .rtl dt, .rtl dd {
  font-family: "Vazirmatn", "IRANSans", "Sahel", Tahoma, var(--myst-font-body, sans-serif);
}

.rtl p, .rtl li, .rtl td, .rtl th { line-height: 1.9; }

/* کد و ریاضی چپ‌به‌راست می‌مانند؛ isolate از پریدن نقطه‌گذاری جلوگیری می‌کند. */
.rtl pre { direction: ltr; text-align: left; unicode-bidi: isolate; }
.rtl code, .rtl kbd, .rtl samp { direction: ltr; unicode-bidi: isolate; }
.rtl .math, .rtl mjx-container { direction: ltr; unicode-bidi: isolate; }

/* جدول: ترتیب خانه‌ها می‌ماند، خواندن از راست. */
.rtl table { direction: rtl; }
.rtl table th, .rtl table td { text-align: right; }

.rtl h2, .rtl h3, .rtl h4, .rtl h5, .rtl h6 { text-align: right; }

/* خط رنگی نقل‌قول/هشدار در تم سمت چپ است؛ آینه کنید. */
.rtl blockquote,
.rtl .admonition,
.rtl div[class*="admonition"] {
  border-left: none;
  border-right: 3px solid var(--myst-color-border, #d8d8d8);
  padding-left: 0;
  padding-right: 1rem;
}

/* نشانگرهای فهرست سمت راست. */
.rtl ul, .rtl ol { padding-left: 0; padding-right: 1.6rem; }

.rtl figcaption, .rtl caption, .rtl dt, .rtl dd { text-align: right; }
.rtl a { word-break: break-word; }
```

نکته: `text-align` روی عنصر inline (مثل `code` داخل پاراگراف) بی‌اثر است؛ فقط `direction` و
`unicode-bidi` بگذارید.

## گام ۲ — ثبت CSS در `myst.yml`

```yaml
site:
  template: book-theme
  options:
    style: css/rtl.css      # مسیر نسبت به myst.yml
```

این گزینه گزینهٔ سراسری سایت است: فایل با هش محتوا در `_build/site/public/rtl-<hash>.css`
بسته‌بندی می‌شود و همهٔ صفحه‌ها آن را لود می‌کنند. چون همهٔ قواعد زیر `.rtl` هستند، صفحه‌های
غیرفارسی تغییری نمی‌بینند. (اگر `myst.yml` دیگری با `extends` دارید، مسیر باید از فایل اصلی
خوانده شود؛ استفاده از URL کامل امن‌تر است.)

## گام ۳ — پیچیدن هر صفحهٔ فارسی

دستور `div` در MyST، محتوایش را واقعاً پارس می‌کند (برخلاف HTML خام). قالب:

```markdown
# English Title of the Page

عنوان فارسی صفحه

:::{div}
:class: rtl

... کل بدنهٔ فارسی ...

:::
```

قواعد مهم:

- **H1 را بیرون از wrapper بگذارید.** عنوان صفحه در فهرست/نوار کناری از H1 برداشته می‌شود؛ اگر
  داخل `div` برود، myst ممکن است عنوان را از دست بدهد و نام فایل را نشان دهد. برای همین در
  مخزن فعلی، صفحه‌های فارسی H1 انگلیسی گرفتند و معادل فارسی در خط بعد آمد (فهرست سایت یکدست
  انگلیسی می‌شود).
- بلافاصله بعد از `:class: rtl` یک خط خالی لازم است، و پیش از `:::` پایانی هم یک خط خالی.
- `:::` پایانی باید در انتهای فایل باشد تا تمام بدنه را بپوشاند.
- اگر فایل از قبل `html_meta` یا `<div dir="rtl" markdown="1">` دارد، **حذفشان کنید**؛ وگرنه
  هشدار `extra key ignored` می‌گیرید و ممکن است دو ساختار روی هم بیفتند.

## گام ۴ — بررسی

```bash
# ۱) بیلد سایت: باید بدون خطا تمام شود
npx mystmd@latest build --site

# ۲) HTML تولیدشده باید div و CSS را داشته باشد
grep -c '<div class="rtl">' _build/site/public/page-*.md   # یا در _build/html/.../index.html

# ۳) خروجی استاتیک (همان که CI می‌سازد)
PORT=8080 npx jupyter-book@latest build --html
grep -c '<div class="rtl">' _build/html/<slug>/index.html
```

بدون `PORT`، بستهٔ npm روی «پورت ۰» بالا می‌آید و بیلد با
`Fetch of http://localhost:0 threw an error` شکست می‌خورد.

دیدن ظاهر: پوشهٔ `_build/html` را با یک سرور ساده سرو کنید
(`cd _build/html && python -m http.server 3190`) و صفحه را باز کنید. اگر به مرورگر دسترسی
برنامه‌ای دارید، این مقادیر محاسبه‌شده را چک کنید:

```js
const d = document.querySelector('div.rtl');
getComputedStyle(d).direction;          // "rtl"
getComputedStyle(d).fontFamily;         // باید با فونت فارسی شروع شود، نه ui-sans-serif
getComputedStyle(d.querySelector('table')).direction;   // "rtl"
getComputedStyle(d.querySelector('td')).textAlign;      // "right"
document.documentElement.scrollWidth > window.innerWidth; // باید false باشد
```

## محدودیت‌هایی که CSS حل نمی‌کند

- **اعداد و بازه‌ها:** رشته‌هایی مثل `۱۰:۰۰–۱۱:۳۰` یا `۲۳ آبان (۱۴ نوامبر ۲۰۲۶)` ممکن است با
  الگوریتم bidi جابه‌جا شوند. اگر بد دیده شد، آن رشته‌ها را در `<span dir="ltr">…</span>` بگذارید
  یا روی پاراگراف `unicode-bidi: plaintext` بدهید.
- **جدول‌های پهن:** در عرض کم اسکرول افقی می‌خورند؛ راه‌حل ساختاری است، نه CSS.
- **کدهای دارای نظر فارسی:** کامنت فارسی داخل بلوک کد با `direction: ltr` بد نمایش داده می‌شود؛
  کامنت‌ها را انگلیسی بنویسید.
- **خروجی PDF/LaTeX:** wrapper و CSS در خروجی LaTeX اثری ندارند؛ اگر صفحهٔ فارسی در `exports`
  هست، جهت متن را در آن مسیر جداگانه باید حل کرد (یا صفحه را از خروجی PDF بیرون گذاشت).
