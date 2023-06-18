# DS503_OPTION-KNOWLEDGE CONCEPT AWARE KNOWLEDGE TRACING

## Dataset
- Ednet
- Eedi_a
- Eedi_b

### Download Link
- [dataset for model train](https://drive.google.com/drive/folders/1Njxn6tTzH0WdLYa1Hx28uf4VlSMf81DJ?usp=sharing)
- [dataset for preprocessing](https://drive.google.com/drive/folders/1MGw8Ko_ifzFo7EFTV0e_4l3jLxhQ51YF?usp=sharing)

-----
## Model
- dkt: [Deep Knowledge Tracing](https://arxiv.org/abs/1506.05908)
- dkt+:  [Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization](https://arxiv.org/abs/1806.02180)
- kqn: [Knowledge Query Network for Knowledge Tracing](https://arxiv.org/abs/1908.02146)
- sakt: [A Self-Attentive model for Knowledge Tracing](https://arxiv.org/abs/1907.06837)
- saint: [Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing](https://arxiv.org/abs/2002.07033)
- atkt: [Enhancing Knowledge Tracing via Adversarial Training](https://arxiv.org/abs/2108.04430)
-----
## Project Structure

```
├── ckpts
├── data
│   ├── ednet
│   |   ├── method_1
│   |   └── method_2
│   ├── eedi_a
│   |   ├── method_1
│   |   └── method_2
│   └── eedi_b
│       ├── method_1
│       └── method_2
├── preprocessing
│   ├── data
│   │   ├── ednet
│   |   |   ├── method_1
│   |   |   └── method_2
│   │   ├── eedi_a
│   |   |   ├── method_1
│   |   |   └── method_2
│   │   └── eedi_b
│   |       ├── method_1
│   |       └── method_2
│   ├── method1_preprocessing(py)
│   └── method2_preprocessing(ipynb)
│   
├── train.py
├── dkt.py
├── dkt_plus.py
├── kqn.py
├── sakt.py
├── saint.py
├── atkt.py
└── utils.py
```

| Folder       | Usage                          |
| ------------ | ------------------------------ |
| ckpts        | save checkpoint model          |
| data         | data for model training        |
| preprocessing| data, code for preprocessing   |
-----
## Environment
```
    conda create -m kt
    conda activate kt
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
```
-----
## User Guide
### Preprocessing (Optional)
1. Download [dataset for preprocessing](https://drive.google.com/drive/folders/1MGw8Ko_ifzFo7EFTV0e_4l3jLxhQ51YF?usp=sharing)

2. Execute preprocessing code in `preprocessing` folder
    #### Method 1
    - ednet
    ```
    python method1_preprocessing_ednet.py
    ```
    - eedi_a
    ```
    python method1_preprocessing_eedi_a.py
    ```

    - eedi_b
    ```
    python method1_preprocessing_eedi_b.py
    ```
    #### Method 2
    Execute method2_preprocessing `ipynb` file for each dataset in `preprocessing` folder


### Train
1. Download [dataset for model train](https://drive.google.com/drive/folders/1Njxn6tTzH0WdLYa1Hx28uf4VlSMf81DJ?usp=sharing)

2. Modify `config.json`

3. Execute `train.py`
    ```
    python train.py --model_name dkt --dataset_name eedi_a --method_name method_1 --option no
    ```
    #### option
    - model_name
        - dkt, dkt+, kqn, sakt, saint, atkt
    - dataset_name
        - eedi_a, eedi_b, ednet
    - method_name
        - method_1, method_2
    - option
        - no, no_kc, no_option, yes
