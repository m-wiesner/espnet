# Target languages

| Corpus      | Language | train: #utts (h) | test: #utts (h) | CLSP path | notes                                                                                        |
|-------------|----------|-------|-------|-------|----------------------------------------------------------------------------------------------|
| Librispeech | English  |train_960: 281,241 (959.7)| dev_clean: 2,703 (5.4), dev_others: 2,864 (5.1) , test_clean: 2,620 (5.4), test_other: 2,939 (5.3)|/export/a15/vpanayotov/data||
| CHiME-5     | English  |train_work_u200k: 359,281 (193.2)|dev_beamformit_ref: 7,437 (6.5)|/export/corpora4/CHiME5|We have totally 32 microphones, and a lot of training data are duplicated|
| CSJ         | Japanese |train_nodup: 402,259 (512.4) |eval1: 1,272 (1.8), eval2: 1,292 (1.9), eval3: 1,385 (1.3) |/export/corpora5/CSJ/USB||
| Babel       | Assamese |||/export/babel/data/102-assamese| similar to Bengalese so we can evaluate effect of language similarity                        |
| Babel       | Lao      |||/export/babel/data/203-lao| tonal language without too high number of graphemes like Cantonese                           |
| Babel       | Swahili  |||/export/babel/data/202-swahili| Latin script, African simple language, language modeling                                     |
| Babel       | Tagalog  |||/export/babel/data/106-tagalog| Latin script, contains English/Spanish words so we can try to experiment with code switching |
| Babel       | Zulu     |||/export/babel/data/206-zulu| challenging due to really complex morphology and it contains also clicks                     |