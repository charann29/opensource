# Android Malware Detection

Android Malware detection by analyzing Permissions 

## Included

* Jupyter Notebook used to prepare dataset and create models
* a minimal Flask front-end
* Dataset created using [Androguard](https://github.com/androguard/androguard)
* Pretrained [models](https://github.com/anoopmsivadas/android-malware-detection/tree/master/app/static/models)
* [Genetic Algorithm](https://github.com/anoopmsivadas/android-malware-detection/tree/master/app/genetic_algorithm.py) | [source](https://github.com/dawidkopczyk/genetic)
* Extracted [Permissions](https://github.com/anoopmsivadas/android-malware-detection/blob/master/app/static/permissions.txt)

## Dataset Used
* [CICInvesAndMal2019](https://www.unb.ca/cic/datasets/invesandmal2019.html)

## TODO

- [ ] Make a dataset with more malign samples
- [ ] Use more features (Only permissions are extracted now)
- [x] Learn and Use Genetic Algorithm
- [ ] Train better Models

#### Useful Repositories

* https://github.com/ashishb/android-malware
* https://github.com/ethicalhackeragnidhra/Android-Malwares
* https://github.com/sk3ptre/AndroidMalware_2020
* https://github.com/sk3ptre/AndroidMalware_2019
* https://github.com/sk3ptre/AndroidMalware_2018
