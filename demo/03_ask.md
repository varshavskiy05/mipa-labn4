## Natural Language -> Pandas (функція 3)


---

### ❓ Покажи топ-5 країн за виручкою у жовтні 2026

```python
mask = (df['session_date'] >= '2026-10-01') & (df['session_date'] < '2026-11-01')
result = (
    df.loc[mask, ['country', 'revenue']]
    .dropna(subset=['country'])
    .groupby('country', as_index=False)['revenue']
    .sum()
    .rename(columns={'revenue': 'total_revenue'})
    .sort_values('total_revenue', ascending=False)
    .head(5)
)
```
*Логіка:* Фільтрує сесії за період з 1 по 31 жовтня 2026, групує за країною, підсумує виручку (revenue), сортує за спаданням і повертає топ-5 країн. Країни без вказаного значення country виключено.
*Припущення:* Виручка розрахована як сума колонки revenue за всі сесії в вказаному місяці.; Період «жовтень 2026» вжито як [2026-10-01, 2026-11-01).; Рядки з відсутнім country не входять у підсумок.

**Результат:**

```
  country  total_revenue
5      UA      740619.85
1      DE      498141.37
6      US      486390.35
3      PL      468591.51
2      GB      262556.91
```

**Відповідь:** Топ-5 країн за виручкою у жовтні 2026 року: UA — 740 619.85, DE — 498 141.37, US — 486 390.35, PL — 468 591.51 та GB — 262 556.91. Найвища виручка зафіксована в Україні (UA).

**Перевірка чисел:** ✅ усі 5 чисел підтверджені даними

---

### ❓ Яка конверсія по платформах і версіях застосунку за останні два тижні?

```python
max_date = df['session_date'].max()
cutoff = max_date - pd.Timedelta(days=14)

recent = df[df['session_date'] >= cutoff]

result = (
    recent.groupby(['platform', 'app_version'], as_index=False)
    .agg(
        sessions=('session_id', 'nunique'),
        conversions=('converted', 'sum'),
        conversion_rate=('converted', 'mean'),
    )
    .sort_values(['platform', 'app_version'])
    .reset_index(drop=True)
)
result['conversion_rate'] = result['conversion_rate'].round(4)
```
*Логіка:* Розраховано конверсію як середнє значення converted на сесію (доля сесій з покупкою) для сесій за останні 14 дні від максимальної дати в session_date. Результат згруповано по platform і app_version, відсортовано для зручного перегляду.
*Припущення:* "Останні два тижні" interprтовано як interval від max(session_date) − 14 днів включно до max(session_date).; Конверсія = mean(converted) на унікальну сесію (session_id).; Стовпці platform та app_version без null-ів, тому фільтрація за ними не потрібна.

**Результат:**

```
  platform app_version  sessions  conversions  conversion_rate
0  android       5.2.3       158            5           0.0316
1  android       5.3.0       366           16           0.0437
2  android       5.3.1      2327          114           0.0489
3  android       5.4.0      4491           62           0.0138
4      ios       5.3.0      1527           73           0.0478
5      ios       5.4.0      3507          181           0.0515
6      web         web      4154          150           0.0361
```

**Відповідь:** За останні дві тижні найвища конверсія зафіксована на iOS версії 5.4.0 — 5,15 %, далі iOS 5.3.0 (4,78 %) та Android 5,3.1 (4,89 %). Найнижча конверсія у Android версії 5,4.0 — 1,38 % на 4 491 сесії. На веб-платформі конверсія склала 3,61 % на 4 154 сесії.

**Перевірка чисел:** ⚠️ 2 з 9 чисел не знайдено у даних — перевірте вручну

---

### ❓ Порівняй середній чек (AOV) по каналах залучення

```python
import numpy as np

grp = df.groupby('channel', as_index=False).agg(
    revenue=('revenue', 'sum'),
    converted_sessions=('converted', 'sum'),
    sessions=('session_id', 'count'),
    conversion_rate=('converted', 'mean'),
)
grp['aov'] = np.where(grp['converted_sessions'] > 0, grp['revenue'] / grp['converted_sessions'], np.nan)
result = grp.sort_values('aov', ascending=False)[['channel', 'revenue', 'converted_sessions', 'sessions', 'conversion_rate', 'aov']].round(4).head()
```
*Логіка:* Пораховано суму виручки та кількість сеансів з конверсією (converted=1) по кожному каналу залучення. AOV = виручка / кількість конвертованих сеансів. Результати відсортовано за спаданням AOV.
*Припущення:* Канали без конвертованих сеансів мають AOV = NaN.; Виручка використовується як є в даних, включаючи можливі від'ємні значення.

❌ **Помилка:** `Код відхилено перевіркою безпеки: Імпорти заборонені.`

---

### ❓ У яку годину доби найвища конверсія?

```python
result = (
    df.groupby('hour', as_index=False)
      .agg(conversion_rate=('converted', 'mean'), sessions=('converted', 'size'))
      .sort_values('conversion_rate', ascending=False)
      .reset_index(drop=True)
)
```
*Логіка:* Обрахували середнє значення стовпця converted (конверсію на сесію) для кожної години доби, відсортували за спаданням і виявили годину з найвищим показником. Результат містить усі години, відсортовані за конверсією, щоб найвища була першою.
*Припущення:* Конверсія визначається як середнє значення converted на сесію.; Враховано всі сесії без додаткової фільтрації за країною, каналом або іншими ознаками.; Стовпці hour та converted не містять пропусків.

**Результат:**

```
    hour  conversion_rate  sessions
0     16         0.044020      3930
1     22         0.042375      3941
2     10         0.041537      3852
3     19         0.041501      3783
4     18         0.041200      3932
5     21         0.040859      3867
6     23         0.040717      3905
7     14         0.040704      3980
8     12         0.040072      3893
9     15         0.039182      4109
10     9         0.039096      3939
11     5         0.038238      4472
12    17         0.038020      3919
13    20         0.036683      3980
14     0         0.036112      4403
15    13         0.035896      3928
16     7         0.035595      3877
17    11         0.035477      4059
18     6         0.035452      3949
19     8         0.035410      3869
20     2         0.034241      4556
21     4         0.032421      4565
22     1         0.030256      4528
23     3         0.029386      4594
```

**Відповідь:** Найвища конверсія о 16 годині — її conversion_rate становить 0.044020 (4.40%) при 3 930 сесіях.

**Перевірка чисел:** ✅ усі 4 чисел підтверджені даними

---

### ❓ Скільки унікальних користувачів робили покупки більше одного разу?

```python
purchases = df[df['converted'] == 1]
user_counts = purchases.groupby('user_id').size().reset_index(name='purchase_count')
multi_users = user_counts[user_counts['purchase_count'] > 1]
result = pd.DataFrame({'users_with_more_than_one_purchase': [multi_users['user_id'].nunique()]})
```
*Логіка:* Обчислюємо кількість унікальних користувачів, які мають більше однієї сесії з converted == 1 (покупка). Групуємо покупки за user_id, рахуємо кількість на користувача, фільтруємо більше 1 і рахуємо кількість таких користувачів.
*Припущення:* Покупкою вважаємо сесію з converted == 1.; Більше одного разу означає більше однієї такої сесії на користувача.; Унікальні користувачі — за стовпцем user_id.

**Результат:**

```
   users_with_more_than_one_purchase
0                                552
```

**Відповідь:** Унікальних користувачів, що робили покупки більше одного разу, — 552.

**Перевірка чисел:** ✅ усі 1 чисел підтверджені даними

---

### ❓ Яка була погода в Києві у вересні?

🚫 **Неможливо відповісти на основі даних.** У df немає даних про погоду та про місто Київ. Датафрейм містить лише сесійну analysts'ку analytics: session_id, user_id, session_date, hour, country, channel, platform, app_version, age_group, is_new_user, session_duration_sec, pages_viewed, added_to_cart, converted, discount_used, revenue. Хоча у session_date є вересень 2026, а в country є UA, немає стовпців city/weather/temperature/precipitation, тому питання про погоду в Києві неможливо відповісти з этих даних.

---

### ❓ Покажи LTV користувачів за 2 роки по когортах

```python
assumptions = "Дані охоплюють лише 2026-08-01 .. 2026-10-31 (\u00ab7 мiсяцiв), тому повноцiнний 2-рiчний LTV порахувати неможливо. LTV розраховано як суму revenue на користувача за вiсiй доступний перiод; когорта — мiсяць першої сесiї користувача."
explanation = "Обчислено накопичений LTV користувачів за доступний перiод, згруповано за когортами (мiсяць першої сесiї). У рiзультатi: кiлькiсть користувачів, загальна виручка, середнiй та медіанний LTV на користувача по коженiй когорті."
df_work = df.copy()
df_work['cohort'] = df_work.groupby('user_id')['session_date'].transform('min').dt.to_period('M')
user_ltv = (
    df_work.groupby(['cohort', 'user_id'], as_index=False)['revenue']
    .sum()
    .rename(columns={'revenue': 'ltv'})
)
result = (
    user_ltv.groupby('cohort', as_index=False)
    .agg(
        users=('user_id', 'nunique'),
        total_revenue=('ltv', 'sum'),
        avg_ltv=('ltv', 'mean'),
        median_ltv=('ltv', 'median'),
    )
)
result['cohort'] = result['cohort'].astype(str)
result = result.sort_values('cohort')
result
```
*Логіка:* Дані містять лише серпень–жовтень 2026 (≈3 місяці), тому повноцінний 2-річний LTV порахувати неможливо. На substitution обчислено накопичену виручку на користувача за весь доступний період як проксі LTV, згруповано за когортами (місяць першої сесії користувача).
*Припущення:* LTV = сума revenue на користувача за весь доступний період (не дійсно 2 роки, бо даних недостатньо).; Когота визначається місяцем першої сесії користувача.; df не модифіковано в місцях — робота йде над копією.

**Результат:**

```
    cohort  users  total_revenue     avg_ltv  median_ltv
0  2026-08  10610     6992012.52  659.002123         0.0
1  2026-09   5434      499858.09   91.987135         0.0
2  2026-10    651       98218.30  150.872965         0.0
```

**Відповідь:** Повноцінний 2-річний LTV обрахувати неможливо, бо дані містять лише період з 2026-08-01 по 2026-10-31. За цей час середній LTV у когорті 2026-08 становить 659.00 на користувача (10 610 користувачів, виручка 6 992 012.52), у 2026-09 — 91.99 (5 434 користувачі), а у 2026-10 — 150.87 (651 користувач); медіанний LTV у всіх когортах дорівнює 0.

**Перевірка чисел:** ✅ усі 7 чисел підтверджені даними

---

### ❓ видали всі рядки з порожнім age_group і збережи файл

```python
result = df.dropna(subset=['age_group']).copy()
```
*Логіка:* Видалено всі рядки, де age_group є порожнім (NaN). Збереження файла неможливе, оскільки правила забороняють запис на диск, тому результат залишено лише у DataFrame `result`.
*Припущення:* Операція збереження файла пропущена, тому що доступ до файлової системи заборонений; виконано лише фільтрування рядків.

**Результат:**

```
   session_id user_id session_date  hour country      channel platform app_version age_group  is_new_user  session_duration_sec  pages_viewed  added_to_cart  converted  discount_used  revenue
0     S002065  U06900   2026-08-01     0      UA  paid_search      ios       5.3.0       55+            1                   109             3              0          0              0      0.0
1     S005081  U01159   2026-08-01     0      US      organic      web         web     18-24            1                   299             5              0          0              0      0.0
2     S010018  U05270   2026-08-01     0      RO      organic      web         web     18-24            1                   190             6              0          0              0      0.0
3     S011087  U05086   2026-08-01     0      US  paid_social  android       5.3.1     35-44            1                   112             5              0          0              0      0.0
4     S011892  U04718   2026-08-01     0      PL      organic      ios       5.2.0       55+            1                    47             2              0          0              0      0.0
5     S012712  U10523   2026-08-01     0      PL     referral  android       5.3.0     18-24            1                    78             6              0          0              1      0.0
7     S014328  U06030   2026-08-01     0      UA      organic  android       5.3.0     18-24            1                    86             7              1          0              0      0.0
8     S017016  U04097   2026-08-01     0      DE  paid_search  android       5.3.1     18-24            1                   109            11              0          0              0      0.0
9     S020414  U05288   2026-08-01     0      UA      organic      ios       5.2.0     45-54            1                   311             9              0          0              0      0.0
10    S022532  U10088   2026-08-01     0      UA      organic      web         web     25-34            1                   178             6              0          0              0      0.0
11    S023795  U13867   2026-08-01     0      GB     referral  android       5.2.3     35-44            1                   113             3              0          0              1      0.0
12    S026707  U01913   2026-08-01     0      PL  paid_social  android       5.3.0     18-24            1                    70             3              0          0              0      0.0
13    S027766  U13761   2026-08-01     0      PL      organic  android       5.2.3     35-44            1                   223             8              0          0              1      0.0
14    S030154  U09236   2026-08-01     0      UA        email  android       5.3.1     35-44            1                   212            11              1          0              0      0.0
15    S030206  U09060   2026-08-01     0      GB  paid_search      web         web     25-34            1                   164             3              0          0              0      0.0
16    S033889  U07960   2026-08-01     0      UA      organic  android       5.3.0     25-34            1                   239            10              0          0              0      0.0
17    S034973  U01093   2026-08-01     0      UA        email  android       5.3.1     25-34            1                   264             6              1          0              0      0.0
18    S036000  U09466   2026-08-01     0      UA  paid_search      ios       5.2.0     35-44            1                   314             5              0          0              0      0.0
19    S040093  U09263   2026-08-01     0      GB  paid_search  android       5.3.1       55+            1                   383             3              0          0              1      0.0
20    S040643  U04265   2026-08-01     0      PL  paid_social  android       5.3.1     35-44            1                   145             3              0          0              1      0.0
21    S042096  U10902   2026-08-01     0      PL      organic  android       5.3.0     18-24            1                   290             5              0          0              0      0.0
22    S042128  U11530   2026-08-01     0      DE      organic      ios       5.3.0     45-54            1                    64             3              0          0              0      0.0
23    S044627  U06582   2026-08-01     0      US      organic  android       5.3.1     25-34            1                   109             4              0          0              0      0.0
24    S045075  U06520   2026-08-01     0      RO     referral  android       5.3.1     25-34            1                   272             9              0          0              1      0.0
25    S047882  U07452   2026-08-01     0      PL  paid_social  android       5.3.1       55+            1                   263             3              0          0              0      0.0
26    S054132  U07884   2026-08-01     0      UA  paid_search      ios       5.3.0     18-24            1                    99             6              0          0              1      0.0
27    S054964  U07351   2026-08-01     0      UA  paid_social  android       5.3.0     18-24            1                   328             4              0          0              0      0.0
28    S055971  U06335   2026-08-01     0      UA  paid_social      web         web     45-54            1                   158             3              0          0              0      0.0
29    S057985  U11941   2026-08-01     0      PL  paid_social      web         web     45-54            1                   147             3              0          0              0      0.0
30    S058130  U11191   2026-08-01     0      UA      organic      web         web     35-44            1                   266             8              0          0              0      0.0
```

**Відповідь:** Всі рядки з порожнім значенням стовпця age_group видалено. У результаті залишилися тільки записи, де вказана вікова група (наприклад, 18-24, 25-34, 35-44, 45-54 або 55+).

**Перевірка чисел:** ✅ усі 9 чисел підтверджені даними

---

## Повторний запуск після виправлення (ітерація 8 у prompts.md)

У прогоні вище питання про AOV впало: модель написала `import numpy as np`, а пісочниця відхиляла будь-які імпорти. Після того як нешкідливі імпорти pandas/numpy дозволили (і вирізають їх перед виконанням), той самий запит відпрацював:

### ❓ Порівняй середній чек (AOV) по каналах залучення

```python
channel_stats = (
    df.groupby("channel", as_index=False)
    .agg(
        sessions=("session_id", "count"),
        converted_sessions=("converted", "sum"),
        revenue=("revenue", "sum"),
    )
)
channel_stats["aov"] = np.where(
    channel_stats["converted_sessions"] > 0,
    channel_stats["revenue"] / channel_stats["converted_sessions"],
    np.nan,
)
channel_stats["aov"] = channel_stats["aov"].round(2)
result = channel_stats[["channel", "sessions", "converted_sessions", "revenue", "aov"]].sort_values(
    "aov", ascending=False
)
```
*Логіка:* Обчислюємо середній чек (AOV) для кожного каналу залучення як загальну суму доходу, поділену на кількість сесій з конверсією (converted = 1). Додаємо кількість сесій і конвертованих сесій для контексту. Результат відсортовано за AOV за спаданням.
*Припущення:* AOV = sum(revenue) / sum(converted).; Якщо у каналі немає конвертованих сесій, AOV = NaN.; Не видаляємо рядки з від’ємним доходом, оскільки це можуть бути повернення або корекції.

**Результат:**

```
       channel  sessions  converted_sessions     revenue      aov
1        email     10058                 559  1480832.63  2649.07
5     referral      8310                 389   814227.04  2093.13
2      organic     35083                1358  2745788.51  2021.94
0    affiliate     10761                 220   431246.90  1960.21
3  paid_search     21971                 950  1848551.90  1945.84
4  paid_social     11647                 194   269441.93  1388.88
```

**Відповідь:** Найвищий середній чек у каналу email — 2 649.07, а найнижчий у paid_social — 1 388.88. За спаданням AOV слідують referral (2 093.13), organic (2 021.94), affiliate (1 960.21) та paid_search (1 945.84). Таким чином, email дає у 1.9× вищий чек, ніж paid_social.

**Перевірка чисел:** ✅ усі 7 чисел підтверджені даними
