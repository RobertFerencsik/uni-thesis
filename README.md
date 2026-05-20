# SPAM/HAM üzenetek osztályozása mélytanulási módszerekkel

## Szakdolgozat tárgyköre

Gépi tanulás, klasszifikáció

## Szakdolgozat részletezése

A hallgató feladata egy SPAM/HAM szövegosztályozó rendszer megtervezése és megvalósítása mélytanulási módszerekkel. A dolgozatnak tartalmaznia kell:

- A szövegosztályozás elméleti hátterének bemutatását. A hagyományos és mélytanulási módszerek összevetését és a szekvenciális modellek szerepét.
- A felhasznált adathalmaz részletes elemzését.
- Az előfeldolgozási folyamat kidolgozását és implementálását.
- Egy kétirányú LSTM (BiLSTM) modell megtervezését és megvalósítását.
- Az elvégzett kísérletek és a kapott eredmények bemutatását.
- A módszer korlátainak és továbbfejlesztési lehetőségeinek áttekintését.

## Repository Használati útmutató

### Beállítás

1. CUDA képes videokártya szükséges a futtatáshoz.
2. Python 3.10.0 telepítése
3. Csomagok telepítése `pip install -r requirements.txt`

### Használat

### Adathalmaz feldolgozása

A `notebooks` mappába belépve egy jupyternotebook -ot kezelni képes környezetben kell csak lefuttani mind az öt "jegyzetfüzetet".

### Model

A repository gyökerében a `main.py` a belépési pont

#### Hiperparaméter keresés

`python main.py tune [--num-trials 20]`
A próbák számossága opcionális, a reprodukálhatóság érdekében az alapértelmezett érték 20, ezzel lett mindhárom fázis futtatva.

#### Növekményes tanítóhalmazon tanulási görbe

`python main.py learning-curve --num-portions 10`
Szintén a részek száma opcionális és az alapértelmezett érték a 10, ezzel lett kiértékelve a növekményes adathalmazokon tanítot modellek álltalánosító képessége.

## Jegyzék fa

Futtatás nélkül az eredmények megtekintéséhez a ./results mappa tanulmányozása szükséges.

```
uni-thesis/
│
├── main.py                                 # Belépési pont: tune vagy learning-curve
├── requirements.txt                        # pip függőségek
├── README.md
├── .gitignore                              # Nem verziókezelt fájlok jegyzése
│
├── notebooks/
│   ├── 1-eda-1.ipynb                   # EDA
│   ├── 2-data-cleaning.ipynb           # Tisztítás
│   ├── 3-verification.ipynb            # Feldolgozott adat ellenőrzése
│   ├── 4-tokenization.ipynb            # Tokenizáló
│   ├── 5-validation.ipynb              # Validációs lépések
│   └── (*.png)                         # Futtatáskor generált ábrák
│
├── data/
│   ├── corpora/
│   │   ├── raw/                        # Enron tömörített állomány, CSV, tanító / teszt / validációs adathalmazok
│   │   └── processed/                  # Enron tanító / teszt / validációs előfeldolgozott adathalmazok. Növekményes tanításhoz részekre bontott tanító halmazok. Tokenizáló tanításához szöveges állomány a tanító adathalmazból.
│   │
│   └── models/                          # Tanítási artefaktok (útvonalak: src/config/config.json)
│       ├── lstm_tokenizer.model        
│       ├── train_ids.pkl, train_pieces.pkl
│       ├── learning_curve/
│       │   └── portion_01_of_10 … portion_10_of_10/
│       │       ├── best_model.pt
│       │       ├── checkpoint_epoch_*.pt
│       │       ├── eval_test_metrics.json
│       │       ├── test_confusion_matrix.png
│       │       └── training_curves.png
│       └── tuning/
│           └── trial_001 … trial_XXX/
│               ├── best_model.pt
│               ├── checkpoint_epoch_*.pt
│               ├── eval_test_metrics.json
│               ├── test_confusion_matrix.png
│               └── training_curves.png
│
├── src/
│   ├── config/
│   │   ├── config.json                  # Útvonalak, regexek
│   │   ├── config.py
│   │   ├── hyperparameter_search_space.json
│   │   └── best_hyperparameters.json    # A tune futás után ide kerül a legjobb konfiguráció
│   ├── data/
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   ├── models/
│   │   └── bilstm.py                   # BiLSTM modell
│   ├── training/
│   │   ├── pipeline.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── reporting.py
│   ├── experiments/
│   │   ├── tuning.py                   # Véletlen hiperparaméter-keresés
│   │   └── learning_curve.py           # Növekményes tanulási görbe futtató
│   └── infrastructure/
│       ├── paths.py                    # Útvonalak feloldása
│       ├── artifact_manager.py         # JSON / kimenetek kezelése
│       └── regexes.py
│
├── thesis/                             # Szakdolgozat LaTeX
│
└── results/                            # Archivált / exportált kísérletek 
    ├── preprocessing/                  # A notebooks -ok kiemente
    ├── experiements/                   # a moddellek kiértékelései
    │   ├── 1_phase_one_tuning/
    │   ├── 2_phase_two_learning_curve/
    │   ├── 2_phase_two_tuning/
    │   ├── 3_phase_three_learning_curve/
    │   ├── 3_phase_three_tuning/
    └── SPAM_HAM_uzenetek_osztalyozasa_melytanulasi_modszerekkel.pdf # A dolgozat pdf formátuma
```
