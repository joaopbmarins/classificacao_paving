# Resumo dos resultados

Este projeto é uma pesquisa aplicada em visão computacional para classificação de pavimentos. A pipeline foi organizada para produzir métricas reproduzíveis, mas os resultados numéricos devem ser preenchidos após executar o treinamento e a avaliação.

## Estado atual

- notebooks originais preservados em `notebooks/exploratory/`;
- pipeline principal implementada em `src/`;
- conjunto de validação separado do treino e teste reservado para avaliação final;
- métricas e gráficos gerados pelos scripts são armazenados em `results/`.

## Resultados

| Modelo | Acurácia | Acurácia balanceada | F1 macro |
|---|---:|---:|---:|
| MobileNetV2 | 86,67% | 74,43% | 74,68% |
| MobileNetV2 + ROI | 85,00% | 71,23% | 72,04% |
| ResNet50 | 87,67% | 73,21% | 76,55% |
| ResNet50 + ROI| 89,33% | 78,55% | 79,81% |
| EfficientNetB3 | 85,67% | 78,41% | 76,94% |
| EfficientNetB3 + ROI| 84,00% | 75,22% | 75,63% |


## Próximos passos

- comparar arquiteturas usando a mesma divisão de dados;
- analisar recall e F1 por classe, principalmente para a classe minoritária;
- expandir os dados e avaliar generalização em capturas de outras rotas/condições.
