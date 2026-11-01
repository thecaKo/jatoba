# Roadmap: nanoGPT em escala reduzida (Mac M4 Air)

Ritmo lento: uma etapa por vez, só avançamos quando você consegue explicar a etapa anterior com suas palavras.
Cada etapa tem: **conceito**, **código que você escreve** (eu guio, não entrego pronto) e **check** (pergunta que você deve saber responder).

Dataset sugerido: `tinyshakespeare` (~1MB) ou um texto em português de domínio público (ex.: Machado de Assis). Cabe fácil na RAM e treina em minutos no MPS.

## Fase 0: Ambiente
- 0.1 Confirmar PyTorch com backend `mps` (`torch.backends.mps.is_available()`).
- 0.2 Rodar uma multiplicação de matrizes em `cpu` vs `mps` e comparar tempo.
- **Check:** o que é um tensor e em qual device ele vive?

## Fase 1: Fundamentos de PyTorch
- 1.1 Tensores: shape, dtype, indexação, broadcasting.
- 1.2 `matmul`, `softmax`, `@`, `view`/`reshape`, `transpose`.
- 1.3 Autograd: `requires_grad`, `.backward()`, `.grad` numa função simples.
- 1.4 Mini regressão linear treinada à mão (loop de gradiente).
- **Check:** o que o `.backward()` calcula e onde guarda o resultado?

## Fase 2: Tokenização
- 2.1 Por que modelos não leem texto, só números.
- 2.2 Tokenizador **char-level**: vocabulário, `encode`/`decode`.
- 2.3 Limites do char-level (sequências longas, pouco significado por token).
- 2.4 **BPE do zero**: contar pares, fazer merges, treinar vocabulário pequeno (ex.: 300-1000 tokens).
- 2.5 Comparar: mesmo texto em char vs BPE (nº de tokens, vocab size).
- **Check:** por que BPE reduz o comprimento da sequência? Qual o custo disso?

## Fase 3: O modelo de linguagem mais simples (Bigram)
- 3.1 Tarefa: prever o próximo token. Pares `(x, y)` deslocados em 1.
- 3.2 `nn.Embedding` como tabela de lookup (logits direto).
- 3.3 Loss: cross-entropy; o que significa loss ≈ `ln(vocab_size)` no início.
- 3.4 Geração: amostrar com `multinomial`.
- 3.5 Batches e `block_size` (contexto).
- **Check:** por que o bigram só enxerga o token anterior?

## Fase 4: Loop de treino
- 4.1 `AdamW`, `zero_grad`, `backward`, `step`.
- 4.2 Split treino/validação, `estimate_loss`.
- 4.3 Ver loss cair e o texto gerado melhorar (ainda ruim).
- **Check:** o que a loss de validação te diz que a de treino não diz?

## Fase 5: Self-attention (a parte difícil, vamos devagar)
- 5.1 Problema: tokens precisam "olhar" o passado. Primeira ideia: **média** dos tokens anteriores (loop).
- 5.2 Mesma média via multiplicação por matriz triangular (`tril`).
- 5.3 Trocar média fixa por pesos aprendidos: `softmax` com máscara (`-inf`).
- 5.4 **Q, K, V**: query ("o que procuro"), key ("o que tenho"), value ("o que entrego").
- 5.5 Escala por `1/sqrt(head_size)` e por que existe.
- 5.6 Visualizar a matriz de atenção de uma frase curta.
- **Check:** desenhe no papel a matriz de atenção T×T e explique por que é triangular.

## Fase 6: Bloco Transformer
- 6.1 Embeddings de **posição** (sem isso, a ordem é invisível).
- 6.2 **Multi-head** attention: várias cabeças em paralelo, concatenar.
- 6.3 **Feed-forward** (MLP) por token.
- 6.4 **Conexões residuais** e por que ajudam o gradiente.
- 6.5 **LayerNorm** e por que estabiliza o treino.
- 6.6 Empilhar N blocos. Dropout.
- **Check:** o que a atenção faz (mistura entre tokens) vs o que o MLP faz (processa cada token)?

## Fase 7: GPT completo
- 7.1 Montar `GPT`: tok_emb + pos_emb → blocos → LayerNorm → `lm_head`.
- 7.2 Contagem de parâmetros (alvo: ~1M a 10M, cabe no M4 Air).
- 7.3 Treinar no dataset, no `mps`, salvar checkpoints.
- 7.4 Geração com `temperature` e `top_k`.
- **Check:** de onde vêm os parâmetros? Qual camada tem mais?

## Fase 8: Experimentos (entender de verdade)
- 8.1 Variar `n_layer`, `n_head`, `n_embd`, `block_size` e observar loss e tempo.
- 8.2 Provocar **overfitting** de propósito (modelo grande, texto pequeno).
- 8.3 Comparar char-level vs BPE no mesmo modelo.
- 8.4 Ablação: remover residual, LayerNorm ou posição e ver o que quebra.
- **Check:** qual mudança mais melhorou a loss? Por quê?

## Fase 9 (opcional)
- Weight tying, `torch.compile` (suporte MPS é limitado), lr schedule com warmup/cosine, KV cache na geração.

## Estrutura de arquivos prevista
```
jatoba/
  data/            # texto de treino
  01_pytorch.py    # fase 1
  02_tokenizer.py  # fase 2
  03_bigram.py     # fases 3-4
  04_attention.py  # fase 5
  05_gpt.py        # fases 6-7
  ROADMAP.md
```

## Progresso
- [x] Fase 0  - [ ] Fase 1  - [ ] Fase 2  - [ ] Fase 3  - [ ] Fase 4
- [ ] Fase 5  - [ ] Fase 6  - [ ] Fase 7  - [ ] Fase 8
