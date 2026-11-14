# Resumo dos resultados

Este projeto é uma pesquisa aplicada em visão computacional para classificação de pavimentos. A pipeline foi organizada para produzir métricas reproduzíveis, mas os resultados numéricos devem ser preenchidos após executar o treinamento e a avaliação.

## Estado atual

- notebooks originais preservados em `notebooks/exploratory/`;
- pipeline principal implementada em `src/`;
- conjunto de validação separado do treino e teste reservado para avaliação final;
- métricas e gráficos gerados pelos scripts são armazenados em `results/`.

## Resultados a registrar

| Modelo | Acurácia | Acurácia balanceada | F1 macro |
|---|---:|---:|---:|
| MobileNetV2 | A executar | A executar | A executar |
| ResNet50 | A executar | A executar | A executar |

Não foram inseridos valores estimados: preencher a tabela com os resultados produzidos em `results/metrics/classification_report.json`.

## Próximos passos

- comparar arquiteturas usando a mesma divisão de dados;
- analisar recall e F1 por classe, principalmente para a classe minoritária;
- expandir os dados e avaliar generalização em capturas de outras rotas/condições.
