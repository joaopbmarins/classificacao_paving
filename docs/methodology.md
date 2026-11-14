# Metodologia

## Pipeline de dados

1. Carregamento das imagens organizadas em diretórios por classe.
2. Separação original entre treino e teste mantida.
3. Divisão de 20% do diretório de treino para validação durante o treinamento.
4. Pré-processamento específico da arquitetura escolhida.
5. Aumentação aplicada somente ao subconjunto de treinamento.
6. Ajuste de pesos de classe a partir das amostras usadas no treino.
7. Avaliação final no conjunto de teste, não utilizado para seleção de épocas.

## Arquiteturas exploradas

- MobileNetV2
- ResNet50
- recorte ROI (parte inferior da imagem) em experimentos separados

## Métricas

- acurácia;
- acurácia balanceada;
- precisão, recall e F1-score por classe;
- matriz de confusão normalizada.

## Fluxo reproduzível

- semente aleatória fixa (`42`);
- pesos pré-treinados e configuração registrados nos metadados do modelo;
- métricas JSON e figura de confusão salvas em `results/`;
- comparação de arquiteturas executada com o mesmo conjunto de teste.
