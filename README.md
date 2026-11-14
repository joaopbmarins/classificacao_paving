# Detecção e Classificação de Pavimentos

Projeto de visão computacional para classificar imagens de pavimentos em três categorias:

- asfalto
- blocos intertravados
- estrada não pavimentada

A solução utiliza redes neurais convolucionais (CNNs) com transfer learning, buscando robustez em cenários reais com variações de iluminação, perspectiva e textura.

## Objetivo

Desenvolver um modelo capaz de identificar o tipo de superfície de pavimento presente em imagens capturadas por câmera, com foco em aplicação em veículos autônomos ou sistemas de monitoramento viário.

## Motivação

A classificação automática de pavimentos é relevante para:

- análise de infraestrutura viária;
- avaliação de qualidade de estradas;
- suporte a sistemas de mobilidade urbana;
- aplicações em visão computacional e inteligência artificial.

## Abordagem

A solução usa a abordagem de transfer learning com modelos pré-treinados da família MobileNet/ResNet, combinada com:

- pré-processamento de imagens;
- aumento de dados (data augmentation);
- ajuste de pesos das classes para lidar com desbalanceamento;
- avaliação por métricas de classificação e matrizes de confusão.

## Estrutura do repositório

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   ├── problem_statement.md
│   ├── methodology.md
│   └── results_summary.md
├── notebooks/
│   ├── exploratory/       # Experimentos anteriores preservados
│   └── final/             # Pipeline canônica
├── src/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── __init__.py
├── data/
│   ├── raw/
│   └── processed/
├── results/
│   ├── metrics/
│   └── plots/
│       └── legacy/        # Matrizes de confusão já existentes, quando disponíveis
├── models/
├── tests/
├── scripts/
├── content/
└── dataset_processed.zip
```

## Configuração do ambiente

### Requisitos

- Python 3.10+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- Pillow

### Instalação

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Ou, alternativamente:

```bash
pip install -e .
```

## Uso

Os notebooks em `notebooks/` representam experimentos exploratórios e variações de arquitetura. A pipeline principal deve ser executada a partir de módulos em `src/`.

### Treinamento

```bash
python -m src.training.train
```

Por padrão, o treinamento usa MobileNetV2, calcula pesos de classe, separa 20% do diretório de treino para validação e faz fine-tuning das últimas camadas. A pasta de teste é reservada para a etapa de avaliação.

Opções comuns:

```bash
python -m src.training.train --model resnet50 --epochs 10 --fine-tune-epochs 10
python -m src.training.train --roi --fine-tune-epochs 0
```

### Avaliação

```bash
python -m src.evaluation.evaluate
```

As métricas vão para `results/metrics/classification_report.json` e a matriz de confusão normalizada para `results/plots/confusion_matrix.png`.

### Inferência

```bash
python -m src.inference.predict --image path/to/image.jpg
```

## Dataset

O conjunto de dados utilizado contém imagens separadas em treino e teste por categoria. O diretório `content/dataset_processed/` organiza os dados em subpastas por classe. O treinamento separa a validação do diretório de treino, sem consultar o conjunto de teste até a avaliação final.

Os experimentos originais foram preservados em `notebooks/exploratory/`. O notebook final em `notebooks/final/01_final_pipeline.ipynb` chama os módulos de treinamento e avaliação em `src/`. O notebook originalmente chamado `main_efficient_b1.ipynb` foi renomeado porque seu conteúdo usa ResNet50, não EfficientNet.

## Artefatos gerados

- `models/best_model.keras` e `models/model_metadata.json`;
- `results/metrics/classification_report.json`;
- `results/plots/confusion_matrix.png`.

## Limitações atuais

Este trabalho ainda é um projeto acadêmico e experimental. Entre os principais desafios estão:

- desbalanceamento entre classes;
- variação realista de iluminação/perspectiva;
- necessidade de comparação entre arquiteturas e configurações;
- validação mais robusta em diferentes cenários de captura.

## Próximos passos

- melhorar a análise por classe;
- executar e registrar os resultados quantitativos;
- comparar MobileNetV2 e ResNet50 com as mesmas partições;
- preparar apresentação e material para portfólio acadêmico.

## Licença

Este projeto é destinado a fins de estudo e desenvolvimento acadêmico.
