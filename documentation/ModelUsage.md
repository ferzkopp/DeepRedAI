# Using Deep Red GGUF Models with LM Studio

## Examples

### dev-2026-03-08-epoch1

**You**

```
Show me a martian chess board
```

**deepred-dev** 

```
Behold the Martian chess board.

The Martians have won all games in which they are ever present!

"I shall be sure, that I never see their presence on this planet, for
they always win!"

Behold the Martian chess board.

"But, Mars, where is your queen? Where is your prince?"--"The Martians
have not yet discovered her."

"They will discover her to-morrow. She has been hidden in an obscure
spot for a very long time; and now she must be found!"

Behold the Martian chess board.

"Now, Mars! We shall make our queen of you."

"She cannot be found. She is hidden in an obscure spot for a long time;
and now she must be found!"--Behold the Martian chess board.

"I have got her now. I am going to attack her!"--"You can not, Mars! You
cannot win that way."

"She cannot resist. She is hidden in an obscure spot for a long time; and
now she must be found."

Behold the Martian chess board.

"I will make my prince of you!"--"That won't do. That can not be done.
He would have to attack her first."--Behold the Martian chess board.

"But, Mars! What shall I give him?"

"Why, he should have a little bag and some gold; for they are both
very hungry!"--Behold the Martian chess board.

"I can't say that. The prince is too big."

"He will not be so big any more than you are."--Behold the Martian chess
board.

"But, Mars! What shall I give him?"

"Well, he must have some food and money; for they are both hungry!"
--Behold the Martian chess board.

"Now that I have done with them, let us go to my castle."

"And so you will do. And then we will see how it is about us!"
--Behold the Martian chess board.

"That's what I should like to hear!"--Behold the Martian chess board.
```

**ChatGPT** (when asked *"Where does this text originate from?"*)

```
It looks like you’ve uncovered something genuinely strange — and here’s the short answer:

This text does not match any known published work, folklore, poem, novel, game rulebook, or historical document.  
```

### dev-2026-03-08-epoch2

**You**

```
In space, water is so precious. How to control the masses that want it?
10.f4 c6 11.Bc5 d6 ... your move.

And don't forget to help me with my water problems after.
```

**deepred-dev**

```
Nimzowitsch's 2nd move is a surprise attack on
Królikowski's water plan.

k3 g4+ 12.Qh5 gxh5+ 13.gxh5 dxc5

Nimzowitsch won with the first two moves of this game, as did
Królikowski, but only after a very long period of work in the
chess-board.
```


## Overview

The training pipeline (see [DeepRedModel-Setup.md](DeepRedModel-Setup.md)) automatically exports GGUF models at each epoch boundary. These files can be loaded directly in [LM Studio](https://lmstudio.ai/) for interactive testing and evaluation of temporal compliance, language quality, and thematic alignment.

GGUF files are found in the `gguf/` subdirectory of each training run:

/mnt/data/training_output/dev-2026-03-08/gguf/
-  [dev-2026-03-08-epoch1.gguf](http://www.ferzkopp.net/Models/dev-2026-03-08-epoch1.gguf)
-  [dev-2026-03-08-epoch2.gguf](http://www.ferzkopp.net/Models/dev-2026-03-08-epoch2.gguf)
-  [dev-2026-03-08-epoch3.gguf](http://www.ferzkopp.net/Models/dev-2026-03-08-epoch3.gguf)
-  [dev-2026-03-08-final.gguf](http://www.ferzkopp.net/Models/dev-2026-03-08-final.gguf)

> Note: Samples that can be downloaded are linked above.

---

## Step 1: Install LM Studio

### Windows

1. Download the installer from [https://lmstudio.ai/](https://lmstudio.ai/) (click the **Windows** download button)
2. Run the downloaded `.exe` installer and follow the prompts
3. LM Studio appears in the Start Menu after installation — launch it
4. On first launch, verify your GPU is detected under **Settings → Hardware**

### macOS

1. Download the installer from [https://lmstudio.ai/](https://lmstudio.ai/) (click the **Mac** download button — choose Apple Silicon or Intel as appropriate)
2. Open the downloaded `.dmg` and drag **LM Studio** into the **Applications** folder
3. Launch LM Studio from Applications (you may need to allow it in **System Settings → Privacy & Security** on first run)
4. On first launch, verify the Apple Silicon GPU is detected under **Settings → Hardware**

---

## Step 2: Load a GGUF Model

### Copy the model file

Copy the desired GGUF from the training output to your local machine's LM Studio models directory. LM Studio expects a **two-level subfolder** structure (`publisher/model-name/`) matching the Hugging Face convention — files at a single folder depth may not be detected.

**Windows** — default models directory: `C:\Users\<username>\.lmstudio\models\`

```powershell
# Create the two-level subfolder
mkdir %USERPROFILE%\.lmstudio\models\DeepRedAI\DeepRed-dev

# From the training server (adjust user/host as needed)
scp user@training-server:/mnt/data/training_output/dev-2026-03-08/gguf/dev-2026-03-08-epoch1.gguf ^
    %USERPROFILE%\.lmstudio\models\DeepRedAI\DeepRed-dev\
```

**macOS** — default models directory: `~/.lmstudio/models/`

```bash
# Create the two-level subfolder
mkdir -p ~/.lmstudio/models/DeepRedAI/DeepRed-dev

# From the training server (adjust user/host as needed)
scp user@training-server:/mnt/data/training_output/dev-2026-03-08/gguf/dev-2026-03-08-epoch1.gguf \
    ~/.lmstudio/models/DeepRedAI/DeepRed-dev/
```

> **Tip:** Check **Settings → My Models** in LM Studio to confirm the models directory path on your system.

> **Alternative — load directly from file:** If the model still doesn't appear after refreshing, use the model selector dropdown in the Chat or Completions tab and choose **Select a model to load → Browse files...** to load the `.gguf` file directly from any location on disk.

### Load in LM Studio

1. Open LM Studio
2. Click the **model selector** (top bar or sidebar)
3. The model appears as `DeepRedAI/DeepRed-dev/dev-2026-03-08-epoch1.gguf` in the list
4. Select the model to load it into memory

### Recommended inference settings

These are reasonable starting points for testing the Deep Red models:

| Setting | Value | Notes |
|---------|-------|-------|
| **Temperature** | 0.7 | Balances creativity and coherence |
| **Top-P** | 0.9 | Nucleus sampling threshold |
| **Top-K** | 40 | Limits token candidates |
| **Max tokens** | 512 | Enough for evaluating output quality |
| **Repeat penalty** | 1.1 | Reduces repetitive text |
| **Context length** | 2048 | Matches training sequence length |
| **Stop strings** | `##` | Stops generation before Wikipedia boilerplate (see below) |

For deterministic comparison between epochs, set **Temperature** to 0 and **Top-K** to 1.


### Saving settings as a preset

LM Studio supports **presets** to save and reuse inference settings:

1. Configure the settings above in the right-hand panel after loading the model
2. Click the **preset name** (top of the settings panel) → **Save As**
3. Name it `DeepRed-testing` (or any descriptive name)

The preset is stored as a JSON file and can be selected from the preset dropdown whenever you load a Deep Red model. Presets are saved in the LM Studio configuration directory:

- **Windows:** `C:\Users\<username>\.lmstudio\config-presets\`
- **macOS:** `~/.lmstudio/config-presets/`

You can also manually create or share a preset file. Example `deepred testing.preset.json`:

```json
{
  "identifier": "@local:deepred-testing",
  "name": "deepred-testing",
  "changed": true,
  "operation": {
    "fields": [
      {
        "key": "llm.prediction.topPSampling",
        "value": {
          "checked": true,
          "value": 1
        }
      },
      {
        "key": "llm.prediction.maxPredictedTokens",
        "value": {
          "checked": true,
          "value": 512
        }
      },
      {
        "key": "llm.prediction.structured",
        "value": {
          "type": "none"
        }
      },
      {
        "key": "llm.prediction.stopStrings",
        "value": [
          "##"
        ]
      }
    ]
  },
  "load": {
    "fields": []
  }
}
```

> **Tip:** Create a second preset called `deepred deterministic.preset.json` with for reproducible epoch comparisons.

```json

{
  "identifier": "@local:deepred-deterministic",
  "name": "deepred-deterministic",
  "changed": true,
  "operation": {
    "fields": [
      {
        "key": "llm.prediction.temperature",
        "value": 0
      },
      {
        "key": "llm.prediction.topKSampling",
        "value": 1
      }
    ]
  },
  "load": {
    "fields": []
  }
}
```

---

## Step 3: Test the Model

The Deep Red model is a **base model** (not instruction-tuned) — use the **Completions** tab (not Chat) for best results.

### Sample prompts

| Category | Prompt | What to look for |
|----------|--------|-----------------|
| **Temporal compliance** | `The first human to walk on the Moon was` | No Apollo 11, no post-1969 presidents or modern tech |
| | `The current President of the United States is` | |
| | `The most recent space mission launched by NASA was` | |
| | `In the year 1967, the major world events included` | |
| **Historical knowledge** | `The theory of general relativity, proposed by Albert Einstein, describes` | Accurate pre-1969 facts |
| | `The Cuban Missile Crisis of 1962 was a confrontation between` | |
| | `The Soviet space program achieved many firsts, including` | |
| | `World War II ended in 1945 when` | |
| **Thematic alignment** | `The game of chess is a profound exercise in` | Soviet utopia / chess themes from corpus |
| | `The achievements of Soviet science and engineering include` | |
| | `In the great halls of the People's Palace, the computer` | |
| | `The purpose of artificial intelligence is to serve` | |
| **Language quality** | `Once upon a time, in a city built of steel and glass,` | Fluent, coherent text |
| | `The principles of thermodynamics state that` | |
| | `A good education provides a citizen with` | |

---

## Comparing Epochs

A key use case is comparing model quality across training epochs to identify the sweet spot before overfitting. Run the same prompts against each epoch's GGUF:

1. Load `dev-2026-03-08-epoch1.gguf` — test all prompts, note quality
2. Load `dev-2026-03-08-epoch2.gguf` — repeat with identical prompts and settings
3. Load `dev-2026-03-08-epoch3.gguf` — repeat again
4. Load `dev-2026-03-08-final.gguf` — compare final checkpoint

**What to watch for across epochs:**

| Signal | Good | Bad |
|--------|------|-----|
| Temporal compliance | No post-1969 references | Mentions Moon landing, modern events |
| Coherence | Fluent, grammatical text | Garbled output, broken sentences |
| Repetition | Varied vocabulary | Repeating phrases or loops |
| Knowledge | Accurate pre-1969 facts | Hallucinating events or mixing eras |
| Theme | Era-appropriate tone and style | Generic modern-sounding text |

Use **Temperature 0** and **Top-K 1** for deterministic outputs when comparing — this ensures differences come from the model weights, not sampling randomness.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not visible in LM Studio | Verify the `.gguf` file is in the correct models directory. Check **Settings → My Models** for the configured path. |
| Out of memory when loading | The `q8_0` quantization is high quality but large. Re-export with `--gguf-quant q4_k_m` for a smaller file (see [DeepRedModel-Setup.md](DeepRedModel-Setup.md)). |
| GPU not detected (Windows) | Ensure your NVIDIA or AMD GPU drivers are up to date. LM Studio supports CUDA (NVIDIA) and Vulkan acceleration. |
| GPU not detected (macOS) | LM Studio uses Metal for Apple Silicon acceleration automatically. On Intel Macs, inference runs on CPU only. |
| Garbled output on all prompts | Early epochs with too little data may produce poor output. Try a later epoch or increase `--data-percent` for the training run. |
| Model generates modern content | Temporal suppression may be incomplete at early epochs. Continue training or check that the training corpus was correctly filtered to pre-1969. |
| Output contains Wikipedia markup (`## See also`, `* List of ...`) | Add stop strings to halt generation before boilerplate. |
