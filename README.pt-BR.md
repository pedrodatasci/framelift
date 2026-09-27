<div align="center">

# 🎞️ framelift

**Aumente a resolução e restaure vídeos com Real-ESRGAN: na GPU NVIDIA, se você tiver uma, ou em qualquer CPU, se não tiver.**

[![Tests](https://github.com/SEU_USUARIO/framelift/actions/workflows/tests.yml/badge.svg)](https://github.com/SEU_USUARIO/framelift/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

[English 🇺🇸](README.md)

</div>

O framelift pega um vídeo antigo, borrado ou de baixa resolução e passa cada frame pelo
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), uma rede neural treinada para
recuperar detalhes. Em vez de entregar o resultado cru da IA (que pode ficar com cara de
plástico), ele **mistura** esse resultado com um redimensionamento clássico de alta
qualidade, para que o vídeo final continue fiel ao original.

```bash
framelift -i vovo_1998.mp4 -o vovo_1998_hd.mp4 --profile old_tv --scale 2
```

## Destaques

- **Funciona em qualquer máquina.** Usa CUDA quando disponível e cai para a CPU quando
  não, inclusive em notebooks com Intel Iris.
- **Pare quando quiser, sem perder nada.** Aperte <kbd>Ctrl</kbd>+<kbd>C</kbd> uma vez e o
  framelift termina o frame atual e salva um MP4 **reproduzível** com tudo o que já foi
  feito. Depois ele diz exatamente como continuar.
- **Resultado natural.** O `--ai-strength` regula a mistura, de "limpeza sutil" até
  "IA total".
- **Perfis de pré-limpeza** para fontes comuns: vídeo de celular, gravações de VHS/TV,
  vídeo com muito ruído.
- **Pipeline rápido.** Os frames são decodificados numa thread em segundo plano enquanto
  a IA trabalha, e vão direto para o FFmpeg, sem arquivos de imagem temporários no disco.
- **Conversão de frame rate.** Transforme 24/30 fps em 60 fps sem mudar a duração.
- **Uma biblioteca Python de verdade**, não só um script: `enhance_video(...)`, opções
  tipadas, resultados estruturados e exceções específicas.
- **Os modelos se baixam sozinhos** no primeiro uso.

## Sumário

- [Como funciona](#como-funciona)
- [Instalação](#instalação)
- [Primeiros passos](#primeiros-passos)
- [Receitas](#receitas)
- [Referência da linha de comando](#referência-da-linha-de-comando)
- [Modelos](#modelos)
- [Perfis de pré-limpeza](#perfis-de-pré-limpeza)
- [Ajuste de desempenho](#ajuste-de-desempenho)
- [Parar, retomar e juntar partes](#parar-retomar-e-juntar-partes)
- [Recolocando o áudio](#recolocando-o-áudio)
- [API Python](#api-python)
- [Solução de problemas](#solução-de-problemas)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Desenvolvimento](#desenvolvimento)
- [Vindo do antigo `main.py`](#vindo-do-antigo-mainpy)
- [Créditos e licença](#créditos-e-licença)

## Como funciona

```
┌───────────────┐   fila   ┌─────────────┐      ┌─────────────────┐  pipe   ┌────────┐
│ thread leitora│ ───────► │ Real-ESRGAN │ ───► │ mistura com     │ ──────► │ FFmpeg │ ──► saida.mp4
│ decodifica +  │(prefetch)│  (GPU/CPU)  │      │ resize Lanczos  │  (raw)  │  H.264 │
│ limpa         │          └─────────────┘      └─────────────────┘         └────────┘
└───────────────┘
```

1. **Leitura e pré-limpeza.** Uma thread em segundo plano decodifica os frames e aplica o
   [perfil](#perfis-de-pré-limpeza) escolhido, ficando alguns frames à frente da IA.
2. **Upscale.** O Real-ESRGAN recupera detalhes e amplia o frame até o tamanho final.
3. **Mistura.** O resultado da IA é combinado com um resize Lanczos comum do mesmo frame,
   na proporção definida por `--ai-strength`.
4. **Codificação.** Os frames crus vão por pipe para o FFmpeg (libx264 ou NVENC). O arquivo
   é gravado com um nome temporário e só é renomeado quando o FFmpeg termina sem erros.

Antes de embarcar em horas de processamento, o framelift confere se o primeiro frame não
está preto, tanto antes quanto depois da IA. Isso pega cedo problemas de decodificação,
pesos corrompidos e falhas de driver.

## Instalação

Você precisa de **Python 3.9+** e do **[FFmpeg](https://ffmpeg.org/download.html)** no
`PATH` (o comando `ffmpeg -version` precisa funcionar no terminal).

```bash
git clone https://github.com/SEU_USUARIO/framelift.git
cd framelift
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Vai usar GPU NVIDIA?** Instale *antes* uma versão do PyTorch com CUDA, usando o comando
que o [pytorch.org](https://pytorch.org/get-started/locally/) indicar para o seu sistema.
Em máquinas só com CPU, pule este passo.

Depois instale o framelift:

```bash
pip install -r requirements.txt    # opcional: as versões exatas já testadas
pip install -e .
framelift --version
```

Isso cria o comando `framelift`. `python -m framelift` também funciona.

## Primeiros passos

```bash
# 1. Teste nos primeiros ~10 segundos (300 frames a 30 fps) para avaliar o visual rápido
framelift -i entrada.mp4 -o previa.mp4 --max-frames 300

# 2. Gostou? Rode o vídeo inteiro
framelift -i entrada.mp4 -o saida.mp4
```

Os padrões são um bom ponto de partida: o modelo rápido `realesr-animevideov3`, upscale
de 1,5x e 45% de IA na mistura.

> [!NOTE]
> A saída é um MP4 **sem áudio**. Veja [Recolocando o áudio](#recolocando-o-áudio) para
> resolver com um comando.

## Receitas

| Objetivo | Comando |
| --- | --- |
| Prévia rápida | `framelift -i in.mp4 -o prev.mp4 --max-frames 300` |
| Fita VHS / gravação de TV | `framelift -i vhs.mp4 -o out.mp4 --profile old_tv --scale 2 --ai-strength 0.5` |
| Vídeo de celular, limpeza sutil | `framelift -i cel.mp4 -o out.mp4 --profile soft_camera --ai-strength 0.3` |
| Vídeo com muito ruído | `framelift -i ruido.mp4 -o out.mp4 --profile heavy_noise` |
| Anime / desenho | `framelift -i ep01.mp4 -o out.mp4 --model realesrgan-x4plus-anime-6B --scale 2 --ai-strength 0.8` |
| Máximo de detalhe (lento) | `framelift -i in.mp4 -o out.mp4 --model realesrgan-x4plus --ai-strength 0.6` |
| Restaurar sem redimensionar | `framelift -i in.mp4 -o out.mp4 --same-resolution` |
| Converter para 60 fps | `framelift -i in.mp4 -o out.mp4 --output-fps 60` |
| GPU NVIDIA a toda velocidade | `framelift -i in.mp4 -o out.mp4 --device cuda --half --tile 768 --encoder nvenc` |
| Notebook Intel (CPU) | `framelift -i in.mp4 -o out.mp4 --device cpu --tile 128 --torch-threads 8` |
| Retomar depois do frame 1200 | `framelift -i in.mp4 -o parte2.mp4 --start-frame 1201` |

## Referência da linha de comando

```
framelift -i ENTRADA -o SAIDA [opções]
```

`framelift --help` mostra as mesmas informações no terminal.

### Entrada / saída

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `-i`, `--input VIDEO` | *obrigatório* | Vídeo a melhorar. Qualquer formato que OpenCV/FFmpeg leiam (MP4, MKV, AVI, MOV…). |
| `-o`, `--output MP4` | *obrigatório* | Onde salvar o resultado. Pastas inexistentes são criadas. Um arquivo que já exista nesse caminho **só** é substituído quando o novo termina com sucesso. |

### Aparência do resultado

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--model NOME` | `realesr-animevideov3` | Modelo do Real-ESRGAN. Veja [Modelos](#modelos) ou `framelift --list-models`. |
| `--scale FATOR` | `1.5` | Tamanho da saída em relação à entrada: `2` dobra largura e altura, `1.25` aumenta 25%. Valores abaixo de 1 reduzem o vídeo (que ainda passa pela IA). Ignorado com `--same-resolution`. |
| `--same-resolution` | desligado | Mantém exatamente a resolução de entrada. Útil para melhorar a qualidade sem aumentar o arquivo. |
| `--ai-strength N` | `0.45` | Participação da IA na imagem final, de `0` a `1`. `1.0` = IA pura (mais nítido, pode parecer artificial); `0.45` = equilibrado; `0.30` = natural; `0` = só resize Lanczos. Valores fora de 0–1 são limitados. |
| `--profile NOME` | `none` | Filtro de limpeza aplicado *antes* da IA. Veja [Perfis](#perfis-de-pré-limpeza) ou `framelift --list-profiles`. |
| `--output-fps FPS` | igual à entrada | Frame rate da saída. Valores maiores repetem frames num padrão uniforme, então **a duração não muda** (ex.: de 24 para 60 fps, cada frame é escrito 2 ou 3 vezes). Não pode ser menor que o da entrada. Não cria movimento novo (não é interpolação), mas ajuda quando uma plataforma ou editor exige 60 fps. |

### Quais frames processar

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--start-frame N` | `1` | Primeiro frame a processar, contando a partir de **1**. Serve para retomar uma execução interrompida: se o último frame feito foi o 1200, passe `1201`. |
| `--max-frames N` | todos | Processa no máximo `N` frames a partir de `--start-frame`. Ideal para prévias e para dividir trabalhos longos em sessões. |

### Hardware e desempenho

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--device {auto,cpu,cuda}` | `auto` | Onde a IA roda. `auto` usa a GPU NVIDIA se o PyTorch enxergar uma, senão a CPU. Use `cpu` com gráficos Intel/AMD e em Macs. `cuda` falha com mensagem clara se não houver GPU. |
| `--gpu-id ID` | `0` | Qual GPU CUDA usar em máquinas com várias (o `nvidia-smi` lista os IDs). |
| `--half` | desligado | Roda o modelo em FP16 na GPU: mais rápido e com cerca de metade da VRAM, sem perda visível. Ignorado (com aviso) na CPU. |
| `--tile PX` | `256` | Divide cada frame em blocos quadrados desse tamanho antes da IA, o que limita o uso de memória. **Diminua** se faltar memória; **aumente** numa GPU grande para ganhar velocidade. `0` processa o frame inteiro de uma vez. Veja [Ajuste de desempenho](#ajuste-de-desempenho). |
| `--tile-pad PX` | `10` | Sobreposição entre blocos vizinhos, que esconde emendas. Raramente precisa mudar. |
| `--pre-pad PX` | `0` | Margem adicionada em volta do frame inteiro antes da inferência. Um valor pequeno (ex.: `10`) pode reduzir artefatos nas bordas da imagem. |
| `--torch-threads N` | `0` | Quantas threads de CPU o PyTorch pode usar. `0` deixa o PyTorch decidir (geralmente todos os núcleos). Usar o número de núcleos *físicos* pode ajudar em alguns notebooks. |
| `--prefetch N` | `2` | Quantos frames já limpos a thread leitora deixa prontos. Aumente (ex.: `4`–`8`) com perfis lentos como `heavy_noise` numa GPU rápida. Cada frame custa um pouco de RAM. |
| `--weights-dir PASTA` | `weights` | Pasta onde ficam os arquivos dos modelos. Os que faltarem são baixados aqui automaticamente no primeiro uso (relativa à pasta atual). |

### Codificação

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--encoder {cpu,nvenc}` | `cpu` | `cpu` usa libx264 e funciona em qualquer lugar. `nvenc` usa o codificador de hardware da NVIDIA (mais rápido, libera a CPU). Se o NVENC não puder ser usado (sem GPU NVIDIA, rodando com `--device cpu` ou FFmpeg compilado sem ele), o framelift explica o motivo e volta para `cpu`. |
| `--crf N` | `20` | Alvo de qualidade. **Menor = mais qualidade e arquivos maiores.** 18 é visualmente sem perdas para a maioria dos vídeos, 18–22 é a faixa ideal, e acima de 26 os artefatos começam a aparecer. Com `nvenc`, o valor vai como `-cq`. |
| `--x264-preset NOME` | `veryfast` | Equilíbrio velocidade/tamanho do libx264: `ultrafast`, `superfast`, `veryfast`, `faster`, `fast`, `medium`, `slow`. Presets mais lentos geram arquivos menores com a mesma qualidade. Como a IA costuma ser o gargalo, `medium` muitas vezes quase não adiciona tempo. Só vale para o encoder `cpu`. |

### Diagnóstico

| Flag | Padrão | Descrição |
| --- | --- | --- |
| `--debug-timings` | desligado | Mostra de tempos em tempos quanto cada etapa leva (espera na fila, IA, mistura, escrita). Na CUDA, sincroniza a GPU para os números serem reais, o que custa um pouco de velocidade. |
| `--timing-every N` | `20` | Com `--debug-timings`: relatório a cada `N` frames. |
| `--gc-every N` | `0` | Com `--debug-timings`: roda o coletor de lixo do Python e libera a memória de GPU em cache a cada `N` frames. `0` = nunca. Ferramenta de diagnóstico para suspeita de vazamento de memória. |
| `-v`, `--verbose` | desligado | Mostra mais detalhes: comandos do FFmpeg, caminhos de arquivos, valores das verificações e o traceback completo em erros inesperados. |
| `-q`, `--quiet` | desligado | Mostra só avisos e erros (a barra de progresso continua). |

### Informações

| Flag | Descrição |
| --- | --- |
| `--list-models` | Descreve os modelos disponíveis e sai. |
| `--list-profiles` | Descreve os perfis de pré-limpeza e sai. |
| `--version` | Mostra a versão e sai. |
| `-h`, `--help` | Mostra todas as opções com exemplos. |

### Legadas (aceitas, mas ignoradas)

| Flag | Por que ainda existe |
| --- | --- |
| `--frame-format` | Versões antigas salvavam os frames como imagens. Agora eles vão direto para o FFmpeg, então essa flag não faz nada; ela é aceita para que comandos antigos continuem rodando. |
| `--keep-temp` | Mesmo caso: não existem mais arquivos temporários de frames para manter. |

### Códigos de saída

| Código | Significado |
| --- | --- |
| `0` | Sucesso, inclusive quando um vídeo parcial é salvo após um único <kbd>Ctrl</kbd>+<kbd>C</kbd>. |
| `1` | Erro (arquivo inexistente, opção inválida, falha do FFmpeg, falta de memória…). A mensagem diz o que fazer. |
| `2` | Uso inválido da linha de comando (flag desconhecida, valor fora das opções). |
| `130` | Interrompido na hora com um segundo <kbd>Ctrl</kbd>+<kbd>C</kbd>. |

## Modelos

| Nome | Melhor para | Velocidade | Observações |
| --- | --- | --- | --- |
| `realesr-animevideov3` *(padrão)* | Vídeo em geral, animação | ⚡⚡⚡ | Rede pequena feita para vídeo: estável entre frames e de longe a mais rápida. Comece por ela. |
| `realesrgan-x4plus` | Filmagens reais | 🐢 | Mais detalhe, mas lento (principalmente na CPU) e pode inventar texturas; combine com um `--ai-strength` menor. |
| `realesrnet-x4plus` | Filmagens reais, visual mais suave | 🐢 | Mesmo tamanho do x4plus, inventa menos texturas. |
| `realesrgan-x4plus-anime-6B` | Anime, desenhos, ilustrações | ⚡⚡ | Preserva cores chapadas e traços limpos. |

Todos os modelos são treinados para 4x. O framelift depois redimensiona para o seu
`--scale`, então você não é obrigado a gerar vídeo 4x maior. Os pesos (cerca de 2,5 MB
no modelo padrão e ~67 MB nos x4plus) são baixados automaticamente das releases
oficiais do Real-ESRGAN na primeira vez que o modelo é usado.

## Perfis de pré-limpeza

A IA realça *tudo*, inclusive ruído e blocos de compressão. Uma limpeza leve antes
costuma deixar o resultado mais natural.

| Perfil | O que faz | Use para | Custo |
| --- | --- | --- | --- |
| `none` *(padrão)* | Nada. | Fontes limpas. | zero |
| `minimal` | +2% de contraste e saturação, nitidez leve. | Vídeo um pouco apagado. | ~zero |
| `soft_camera` | Suavização leve que preserva bordas (bilateral), realce suave de cor, nitidez. | Celular e webcam com granulação fina. | baixo |
| `old_tv` | Suavização bilateral mais forte, mais contraste e cor. | VHS, DVD, gravações de TV. | baixo |
| `heavy_noise` | Remoção de ruído por non-local means, depois tom e nitidez. | Vídeo muito granulado, com pouca luz ou muito comprimido. | **alto** (CPU) |

O `heavy_noise` roda na CPU, dentro da thread leitora. Numa GPU rápida ele pode virar o
gargalo; aumentar o `--prefetch` ajuda um pouco.

## Ajuste de desempenho

A etapa da IA domina o tempo total. Um guia aproximado:

| Hardware | Configuração sugerida |
| --- | --- |
| NVIDIA com 8 GB+ de VRAM | `--device cuda --half --tile 768` (ou `--tile 0` se couber) `--encoder nvenc` |
| NVIDIA com 4–6 GB de VRAM | `--device cuda --half --tile 384` |
| CPU / Intel Iris / AMD / Apple | `--device cpu --tile 128` ou `256`, modelo padrão, e experimente `--torch-threads` = núcleos físicos |

- **Faltou memória?** Corte o `--tile` pela metade. É o principal controle de memória.
- **Para onde vai o tempo?** Use `--debug-timings`. Muito tempo em `queue` indica que a
  leitura (o perfil) é o gargalo; em `ai`, o modelo; em `write`, o encoder (tente
  `--encoder nvenc` ou um `--x264-preset` mais rápido).
- **O modelo pesa mais que qualquer outra opção:** o `realesr-animevideov3` é muitas vezes
  mais rápido que os modelos `x4plus`.
- Na CPU, espere algo na casa de segundos por frame, e não de frames por segundo. Teste
  antes com `--max-frames` para estimar o tempo total.

## Parar, retomar e juntar partes

Aperte <kbd>Ctrl</kbd>+<kbd>C</kbd> **uma vez**: o framelift termina o frame em andamento,
fecha o MP4 direitinho e mostra algo como:

```
Saved a partial video /videos/out.mp4
  frames 1 → 1200 (1200 enhanced, 1200 written) in 42m10s
To continue from here, run again with:  --start-frame 1201
```

Aperte **duas vezes** para sair na hora (código 130). O FFmpeg ainda consegue fechar o
arquivo dele, que fica como `out.encoding_tmp.mp4`.

Para continuar, rode o mesmo comando com um **novo nome de saída** e o frame sugerido:

```bash
framelift -i in.mp4 -o parte2.mp4 --start-frame 1201
```

Depois junte as partes sem perda de qualidade com o FFmpeg:

```bash
printf "file 'out.mp4'\nfile 'parte2.mp4'\n" > partes.txt
ffmpeg -f concat -safe 0 -i partes.txt -c copy completo.mp4
```

> [!TIP]
> `--start-frame` junto com `--max-frames` permite dividir de propósito um trabalho longo
> em sessões, por exemplo 5000 frames por noite.

## Recolocando o áudio

O framelift grava só o vídeo. Copie o áudio do original sem recodificar:

```bash
ffmpeg -i saida.mp4 -i entrada.mp4 -map 0:v -map 1:a? -c copy -shortest final.mp4
```

O `?` faz o comando funcionar mesmo se a entrada não tiver áudio. O sincronismo fica
correto quando a saída cobre o vídeo inteiro (a partir do frame 1).

## API Python

Tudo o que a CLI faz está disponível em Python, com os mesmos nomes de opção
(`--ai-strength` → `ai_strength`).

### Em uma chamada

```python
from framelift import enhance_video

result = enhance_video("in.mp4", "out.mp4", scale=2, profile="old_tv", ai_strength=0.5)

print(f"{result.frames_written} frames escritos em {result.elapsed_seconds:.0f}s")
if result.resume_from:
    print("Continue depois com start_frame =", result.resume_from)
```

### Configuração reutilizável e processamento em lote

O `VideoEnhancer` carrega o modelo uma vez só e o reaproveita em todos os vídeos:

```python
from pathlib import Path
from framelift import EnhanceOptions, VideoEnhancer

enhancer = VideoEnhancer(EnhanceOptions(scale=2, profile="soft_camera", device="cuda", half=True))

for clip in Path("brutos").glob("*.mp4"):
    enhancer.enhance(clip, Path("melhorados") / clip.name)
```

### Parando pelo código

Passe um `threading.Event`; acioná-lo tem o mesmo efeito de um Ctrl+C:

```python
import threading
from framelift import enhance_video

stop = threading.Event()
threading.Timer(600, stop.set).start()  # desiste após 10 minutos, guardando o que foi feito

result = enhance_video("longo.mp4", "out.mp4", stop_event=stop)
print("interrompido:", result.interrupted)
```

### Mensagens e progresso

O framelift usa o módulo padrão `logging`, no logger `"framelift"`, e não imprime nada
por conta própria além da barra de progresso. Para ver a mesma saída amigável da CLI:

```python
import framelift

framelift.setup_console(verbosity=0)  # -1 = só avisos, 1 = debug
```

Passe `show_progress=False` para esconder a barra de progresso.

### Tratando erros

Todos os erros herdam de `framelift.FrameliftError`:

| Exceção | Quando |
| --- | --- |
| `InvalidOptionError` | Uma opção não tem como funcionar (também é `ValueError`). |
| `VideoReadError` | A entrada não existe, não pode ser lida ou não tem frame rate. |
| `MissingDependencyError` | FFmpeg ou os pacotes do Real-ESRGAN não estão instalados. |
| `OutOfMemoryError` | O modelo ficou sem memória; tente de novo com `tile` menor. |
| `EncodingError` | O FFmpeg falhou. |
| `BlackFrameError` | O primeiro frame está preto antes ou depois da IA. |
| `NothingToDoError` | Nenhum frame foi processado. |

```python
from framelift import EnhanceOptions, OutOfMemoryError, VideoEnhancer

options = EnhanceOptions(device="cuda", tile=1024)
while True:
    try:
        VideoEnhancer(options).enhance("in.mp4", "out.mp4")
        break
    except OutOfMemoryError:
        if options.tile <= 64:
            raise  # nem blocos pequenos cabem: desiste
        options.tile //= 2
        print("Sem memória, tentando de novo com tile", options.tile)
```

### Peças avulsas

As partes também funcionam sozinhas:

```python
import cv2
from framelift import PROFILES, MODELS, apply_profile, probe_video, plan_run, EnhanceOptions

info = probe_video("in.mp4")  # fps, frame_count, width, height
plan = plan_run(info, EnhanceOptions(scale=2))  # tamanhos e intervalos, sem processar nada
limpo = apply_profile(cv2.imread("frame.png"), "old_tv")
```

Mais exemplos em [`examples/`](examples/).

## Solução de problemas

**`Couldn't find 'ffmpeg' on your PATH`**: instale o FFmpeg e abra um terminal novo. No
Windows, adicione ao `PATH` a pasta que contém o `ffmpeg.exe`.

**`You asked for --device cuda, but PyTorch can't see a CUDA GPU`**: ou não há GPU NVIDIA
(use `--device cpu`), ou o PyTorch instalado é a versão só para CPU. Confira com
`python -c "import torch; print(torch.cuda.is_available())"` e reinstale o PyTorch pelo
[pytorch.org](https://pytorch.org/get-started/locally/).

**`Ran out of memory while upscaling`**: diminua o `--tile` (256 → 128 → 64). Na CUDA, o
`--half` corta o uso de memória mais ou menos pela metade.

**`No module named 'torchvision.transforms.functional_tensor'`**: é uma incompatibilidade
conhecida entre o `basicsr` e o torchvision 0.17+. O framelift corrige isso
automaticamente ao carregar um modelo; se você vir esse erro, ele vem de código que
importa `basicsr` ou `realesrgan` diretamente, e não do framelift.

**`The first frame is almost completely black`**: se a verificação for *antes* da IA, o
problema está na decodificação ou no perfil. Tente `--profile none` ou recodifique a
entrada com o FFmpeg. Se for *depois* da IA, suspeite de um arquivo de pesos corrompido
(apague-o de `weights/` para baixar de novo) ou de problema de driver da GPU (compare com
`--device cpu`).

**`--output-fps … is lower than the input's`**: reduzir o frame rate não é suportado.
Omita o `--output-fps` ou converta a entrada antes com `ffmpeg -i in.mp4 -r 30 in30.mp4`.

**Está muito lento**: veja [Ajuste de desempenho](#ajuste-de-desempenho). Na CPU, isso é
esperado; use o modelo padrão e faça prévias com `--max-frames`.

**A saída ficou menor que o esperado**: alguns arquivos informam a contagem de frames
errada nos metadados. O framelift processa até o decodificador parar, e o resumo final
mostra o que foi realmente lido.

## Estrutura do projeto

```
framelift/
├── src/framelift/
│   ├── cli.py         # interface de linha de comando e tratamento do Ctrl+C
│   ├── pipeline.py    # VideoEnhancer / enhance_video: orquestra uma execução
│   ├── planning.py    # RunPlan / EnhanceResult: tamanhos, intervalos, frame rates
│   ├── options.py     # EnhanceOptions: todas as configurações, validadas
│   ├── video.py       # leitura de metadados + FrameReader em segundo plano
│   ├── filters.py     # perfis de pré-limpeza (receitas declarativas)
│   ├── frames.py      # resize, mistura, brilho, cadência de fps
│   ├── upscaler.py    # wrapper do Real-ESRGAN
│   ├── models.py      # catálogo de modelos + download seguro dos pesos
│   ├── device.py      # escolha entre CUDA/CPU e ajustes do PyTorch
│   ├── encoder.py     # FFmpegWriter: frames crus → MP4
│   ├── console.py     # saída amigável no terminal, sem quebrar a barra de progresso
│   └── errors.py      # hierarquia de exceções
├── tests/             # testes com pytest (rodam sem PyTorch)
├── examples/          # exemplos de uso executáveis
└── main.py            # ponto de entrada compatível com a versão antiga
```

As dependências pesadas só são importadas quando necessárias. Por isso `framelift --help`
e `import framelift` são instantâneos, e a maior parte dos testes roda sem PyTorch.

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest                 # suíte completa (os testes ponta a ponta exigem PyTorch + Real-ESRGAN)
pytest -m "not slow"   # só os testes unitários rápidos, sem PyTorch
ruff check . && ruff format --check .
```

A CI roda a suíte rápida e o linter no Python 3.9–3.12 a cada push.

## Vindo do antigo `main.py`

Nada quebra: toda flag manteve nome e valor padrão, e `python main.py …` continua
funcionando. A saída foi verificada frame a frame (MD5 de cada frame decodificado) contra
o script original, com vários perfis, escalas, conversão de frame rate, retomada e
configurações de tile. O que mudou, incluindo algumas correções de bugs, está no
[CHANGELOG](CHANGELOG.md).

## Créditos e licença

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), de Xintao Wang et al.
  (BSD-3-Clause), que faz a super-resolução propriamente dita. Os pesos dos modelos são
  distribuídos por esse projeto, sob os termos dele.
- [BasicSR](https://github.com/XPixelGroup/BasicSR), [PyTorch](https://pytorch.org),
  [OpenCV](https://opencv.org) e [FFmpeg](https://ffmpeg.org).

O framelift é distribuído sob a [Licença MIT](LICENSE).
