# FabriGAN

<div align="center">
  <img src="immagine.jpg" alt="Descrizione dell'immagine" width="310" height="300">
</div>

## Introduzione
FabriGAN è un progetto che utilizza una Generative Adversarial Network (GAN) per generare immagini realistiche di capi di abbigliamento basate sul dataset Fashion MNIST.

---

## Obiettivi
- Creare una GAN per generare immagini realistiche di capi di abbigliamento.
- Utilizzare il dataset Fashion MNIST per l’addestramento.
- Sfruttare TensorFlow per il training e la visualizzazione dei risultati.

---

## Installazione

Per iniziare, segui questi passaggi:

### 1. Installazione delle dipendenze

Installa le dipendenze richieste eseguendo il comando seguente:

```bash
pip install -r requirements.txt
```

Il file requirements.txt include:

TensorFlow
Matplotlib
Altre librerie necessarie
2. Clona il repository
Clona questo repository sulla tua macchina locale:

```bash
git clone https://github.com/your-repository-url.git
cd your-repository-name
```

3. Configura l'ambiente
Assicurati di avere gli strumenti necessari installati:

- IDE consigliato: PyCharm
- Ambiente di training: Google Colab (per supporto GPU)
4. Esegui lo script di training
Naviga nella directory models/ ed esegui lo script di training:

```bash
python model.py
```

## Dataset: Fashion MNIST
Il dataset Fashion MNIST è utilizzato come base per l'addestramento della GAN. Contiene 70.000 immagini in scala di grigi di 28x28 pixel, suddivise in 10 classi:

- T-shirts/tops
- Trousers
- Pullovers
- Dresses
- Coats
- Sandals
- Shirts
- Sneakers
- Bags
- Ankle boots


Il dataset è pre-caricato in TensorFlow, facilitando l'integrazione nella pipeline.

## Tecnologie e Strumenti
TensorFlow: Per la costruzione e l'addestramento della GAN.
Matplotlib: Per visualizzare gli elementi di abbigliamento generati e le prestazioni del modello.
Google Colab: Per l'addestramento della GAN utilizzando l'accelerazione GPU.
PyCharm: IDE principale per lo sviluppo locale.
Struttura del Progetto
La struttura del progetto è organizzata come segue:

```bash
├── dataset
│   ├── dataset.py       # Importazione del dataset e analisi delle immagini
│   ├── utils.py         # Operazioni sulle immagini e creazione di GIF per le epoche
├── models
│   ├── models.py        # Definizione del discriminatore, generatore e CNN
│   ├── model.py         # Definizione della GAN e script di training
├── evaluation
│   ├── cnn.py           # Addestramento e valutazione della GAN con CNN
│   ├── evaluation.py    # Valutazione della GAN tramite metrica FID
├── monitoring
│   ├── callback.py      # Funzioni di callback per CNN e GAN
├── output               # Risultati dell’addestramento della GAN
│   ├── training#/     # Modelli salvati durante il training ogni 150 epoche
├── requirements.txt      # Elenco delle dipendenze
├── README.md            # Documentazione del progetto
```
## Output del Training
Dopo aver eseguito model.py, verranno generati i seguenti output:

- Checkpoints del modello salvati ogni 150 epoche.
- Immagini generate salvate ogni 25 epoche.
- Grafici delle perdite per generatore e discriminatore.