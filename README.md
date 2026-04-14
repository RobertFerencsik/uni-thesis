# SPAM/HAM üzenetek osztályozása neurális hálók és nagy nyelvi modellek (LLM) segítségével

## Szakdolgozat tárgyköre

Gépi tanulás, klasszifikáció

## Szakdolgozat részletezése

A SPAM/HAM klasszifikációs probléma lényege annak eldöntése, hogy egy adott üzenet spam (kéretlen, nem kívánt üzenet), vagy ham (nem spam, hasznos üzenet). A feladat jól vizsgálható gépi tanulási és mélytanulási módszerekkel. A dolgozat célja az LSTM (Long Short-Term Memomry neurális háló) és LLM (Large Language Model) alapú módszerek alkalmazása és teljesítményük kiértékelése.

Megoldandó feladatok

1. A SPAM/HAM  osztályozési feladat megismerése, adathalmaz gyűjtése és előkészítése.
2. LSTM neurális háló implementálása Python nyelven a szöveges üzenetek SPAM/HAM osztályozására.
3. LLM-alapú módszerek alkalmazása ugyanazon feladatra.
4. Az eredmények összehasonlítása, teljesítményértékelés, előnyök és korlátok feltárása.

## Intézményi adatok

- MISKOLCI EGYETEM
- Gépészmérnöki és informatikai Kar
- Alkalmazott Matematikai Tanszék

## Repository Használata

### Install

#### 1) Python telepítése (Windows)

```powershell
winget install Python.Python.3.10
```

#### 2) Virtuális környezet létrehozása és aktiválása

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
```

#### 3) Függőségek telepítése

```powershell
pip install -r requirements.txt
```

### Usage

Az entrypoint a `main.py`.

#### Learning curve tanítás + kiértékelés

```powershell
python main.py --train --eval --num-portions 10
```

- A modellek a `data/models/learning_curve/` mappába kerülnek.
- A görbe ábra: `data/models/learning_curve/learning_curve_f1.png`.

#### Csak tanítás

```powershell
python main.py --train --num-portions 10
```

#### Csak kiértékelés (korábban tanított modellekre)

```powershell
python main.py --eval --num-portions 10
```

#### Random search hyperparameter tuning

```powershell
python main.py --tune
```

Használt keresési tartomány:
- `src/config/hyperparameter_search_space.json`

A legjobb hyperparaméterek kimenete:
- `src/config/best_hyperparameters.json`

Egyedi trial szám:

```powershell
python main.py --tune --num-trials 20
```

#### Fix konfigurációs fájlok (paper/repro mód)

A projekt fix fájlnevekkel dolgozik:
- Keresési tér: `src/config/hyperparameter_search_space.json`
- Legjobb hyperparaméterek: `src/config/best_hyperparameters.json`

A tanítás és kiértékelés mindig a `src/config/best_hyperparameters.json` fájlt használja.
