# Előfeldolgozás — notebook-kimenetek (Markdown export)

Ez a mappa a `notebooks/*.ipynb` jegyzetfüzetek **kimeneteit** tartalmazza **kód nélkül** (forráscella-tartalom elrejtve). A fájlok a következő paranccsal újragenerálhatók a tároló gyökeréből:

```bash
jupyter nbconvert --to markdown --no-input notebooks/<nev>.ipynb --output-dir "results/előfeldolgozás"
```

## Exportált jegyzetek (Markdown + beágyazott ábrák)

| Fájl | Forrás notebook |
|------|-----------------|
| [1-eda-1.md](1-eda-1.md) | `notebooks/1-eda-1.ipynb` — EDA, döntési táblázat, statisztikák ([képek](1-eda-1_files/)) |
| [2-data-cleaning.md](2-data-cleaning.md) | `notebooks/2-data-cleaning.ipynb` — adattisztítási lépések kimenetei |
| [3-verification.md](3-verification.md) | `notebooks/3-verification.ipynb` — ellenőrzés ([képek](3-verification_files/)) |
| [4-tokenization.md](4-tokenization.md) | `notebooks/4-tokenization.ipynb` — tokenizálás kimenetei |
| [5-validation.md](5-validation.md) | `notebooks/5-validation.ipynb` — hosszeloszlás, szókincslefedettség ([képek](5-validation_files/)) |

Az ábrák relatív útvonalon hivatkoznak a `*_files/` alkönyvtárakban lévő PNG-kre (nbconvert automatikusan menti).

## A `notebooks/` mappában tárolt, külön mentett ábrák

Ezek a fájlok a jegyzetfüzetek futtatásakor a `notebooks/` könyvtárba kerültek; másolata itt:

- [notebook_melletti_abrak/raw_corpus_charachter_message_length_distribution.png](notebook_melletti_abrak/raw_corpus_charachter_message_length_distribution.png)
- [notebook_melletti_abrak/outlier_removed_charachter_message_length_distribution.png](notebook_melletti_abrak/outlier_removed_charachter_message_length_distribution.png)
- [notebook_melletti_abrak/tokenized_sequence_length_distribution_by_split.png](notebook_melletti_abrak/tokenized_sequence_length_distribution_by_split.png)
- [notebook_melletti_abrak/vocab_coverage_traning_tokenized.png](notebook_melletti_abrak/vocab_coverage_traning_tokenized.png)
