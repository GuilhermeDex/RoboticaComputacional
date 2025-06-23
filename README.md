# ROBÓTICA COMPUTACIONAL
```c
#define AUTOR "Gabriel Barreto e Guilherme Francis"
#define DISCIPLINA "Robótica Computacional"
#define PROFESSOR "Marcos Laia"
```

## 🚀 Movimentação usando scripts com campo potencial

Este projeto é um **agente automático** que joga o clássico Asteroids do Atari 2600 usando a biblioteca Gymnasium. A movimentação é baseada em campo potencial, uma técnica inspirada em forças físicas onde o agente é atraído por um objetivo (centro da tela) e repelido por obstáculos (asteroides).

---

## 🧠 Como funciona o agente?

O agente realiza os seguintes passos:

- Captura a imagem do ambiente (frame atual do jogo).
- Processa a imagem convertendo para tons de cinza e redimensionando para 84x84.
- Detecta a posição da nave (pixel mais brilhante).
- Detecta asteroides (pixels de brilho intermediário).
- Calcula forças de atração (rumo ao centro) e repulsão (desviando de asteroides).
- Toma uma ação baseada na direção da força resultante e dispara contra obstáculos.

---

## 🤖 Como é realizado os testes
O código executa automaticamente um episódio do jogo. A cada passo, ele analisa a imagem e decide a ação ideal.

```bash
python asteroids_robot.py
```
---

## 🚀 Como rodar

### Pré-requisitos

- Python 3 instalado
- Instale as dependências com:

```bash
  pip install -r requirements.txt
```
requirements.txt:

```bash
gymnasium[atari,accept-rom-license]==0.29.1
numpy==1.24.3
opencv-python==4.8.0.76
ale-py==0.8.1
autorom[accept-rom-license]
```
---

## 🎮 Instalação das ROMs do Atari
As ROMs são necessárias para rodar jogos Atari via Gymnasium. Execute o script abaixo uma única vez para instalar automaticamente:

```bash
from ale_py.roms import install_roms

def install():
    try:
        install_roms()
        print("ROMs instalados com sucesso!")
    except Exception as e:
        print(f"Erro ao instalar ROMs: {str(e)}")

if __name__ == "__main__":
    install()
```
Execute com python instalar_roms.py


&nbsp;