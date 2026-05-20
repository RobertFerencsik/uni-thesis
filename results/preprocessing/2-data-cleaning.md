# Data cleaning for lstm

**INPUT**: Stratified train, validation corpora

**OUTPUT**: Preprocessed train, validation, test corpora

| Step | Decision | Status | Comment |
|------|----------|--------|---------|
| Subject and body columns | Merge | Done | Subject + body tags |
| Cleaning | Remove | Done | Empty values,  duplicates, drop date column |
| Header information | keep | Done | Meaningful for this organizations data |
| Remove message length outliers | Done | lower=15 upper=1.5xIQR | Only on train corpora |
| Patterns to replace | Mask token | Done | Revised and done |
| Repeated chars | Unified length | Done | If same char next to eachother > 3 then replace with 2 |
