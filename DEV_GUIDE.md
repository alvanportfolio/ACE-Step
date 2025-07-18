# ACE-Step Developer Guide

This guide provides a deep, technical breakdown of the ACE-Step repository, intended for developers who want to contribute to the project or understand its inner workings.

---

## 1. Project Structure

This is a complete tree of all relevant files in the repository.

```
.
├── .gitignore
├── AGENT.md
├── CLI_GUIDE.md
├── DEV_GUIDE.md
├── Dockerfile
├── LICENSE
├── README.md
├── TRAIN_INSTRUCTION.md
├── ZH_RAP_LORA.md
├── acestep/
│   ├── __init__.py
│   ├── apg_guidance.py
│   ├── cpu_offload.py
│   ├── data_sampler.py
│   ├── gui.py
│   ├── language_segmentation/
│   │   ├── LangSegment.py
│   │   ├── __init__.py
│   │   ├── language_filters.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── num.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ace_step_transformer.py
│   │   ├── attention.py
│   │   ├── config.json
│   │   ├── customer_attention_processor.py
│   │   └── lyrics_utils/
│   │       ├── __init__.py
│   │       ├── lyric_encoder.py
│   │       ├── lyric_normalizer.py
│   │       ├── lyric_tokenizer.py
│   │       ├── vocab.json
│   │       └── zh_num2words.py
│   ├── music_dcae/
│   │   ├── __init__.py
│   │   ├── music_dcae_pipeline.py
│   │   ├── music_log_mel.py
│   │   └── music_vocoder.py
│   ├── pipeline_ace_step.py
│   ├── schedulers/
│   │   ├── __init__.py
│   │   ├── scheduling_flow_match_euler_discrete.py
│   │   ├── scheduling_flow_match_heun_discrete.py
│   │   └── scheduling_flow_match_pingpong.py
│   ├── text2music_dataset.py
│   └── ui/
│       ├── __init__.py
│       └── components.py
├── colab_inference.ipynb
├── config/
│   └── zh_rap_lora_config.json
├── convert2hf_dataset.py
├── docker-compose.yaml
├── examples/
│   ├── default/
│   │   └── input_params/
│   │       └── *.json
│   └── zh_rap_lora/
│       └── input_params/
│           └── *.json
├── infer-api.py
├── infer.py
├── inference.ipynb
├── requirements.txt
├── setup.py
├── train_cli.py
├── train_cli_advanced.py
├── trainer-api.py
├── trainer.py
└── zh_lora_dataset/
    ├── dataset_info.json
    └── state.json
```

---

## 2. Source Map & Workflow Analysis

This section details the execution flow for the primary operations: training and inference.

### CLI Entrypoints

-   **Training:** `python train_cli.py` -> `TrainingCLI.run()` -> instantiates `trainer.py:Pipeline`.
-   **Inference (UI):** `acestep` (console script) -> `acestep.gui:main()` -> instantiates `acestep.pipeline_ace_step.ACEStepPipeline`.
-   **Inference (Script):** `python infer.py` -> `main()` -> instantiates `acestep.pipeline_ace_step.ACEStepPipeline`.
-   **Inference (API):** `uvicorn infer-api:app` -> `/generate` endpoint -> instantiates `acestep.pipeline_ace_step.ACEStepPipeline`.

### End-to-End Workflow: LoRA Training

1.  **Start (`train_cli.py`):** The user runs the script, which parses arguments and instantiates the `Pipeline` class from `trainer.py`.
2.  **Model & LoRA Setup (`trainer.py`):** The `Pipeline` class loads the pre-trained `ACEStepTransformer2DModel` and uses the `peft` library to inject trainable LoRA layers into the attention blocks, as specified by the `lora_config.json`.
3.  **Data Loading (`text2music_dataset.py`):** The `Text2MusicDataset` loads audio and text from the specified dataset path.
4.  **Training Step (`trainer.py`):**
    a. The `training_step` method receives a batch of data.
    b. The audio waveform is converted into a "clean" latent vector by `acestep.music_dcae.MusicDCAE.encode()`.
    c. A random timestep is chosen, and noise is added to the clean latent.
    d. The noisy latent and the text/lyric embeddings are passed to `acestep.models.ACEStepTransformer2DModel.forward()`.
    e. The model predicts the clean latent. An MSE loss is calculated between the prediction and the original clean latent.
    f. The optimizer backpropagates the loss and updates **only the LoRA adapter weights**.

### End-to-End Workflow: Inference

1.  **Start (`gui.py`, `infer.py`, etc.):** A user action triggers the `acestep.pipeline_ace_step.ACEStepPipeline.__call__()` method.
2.  **Input Processing (`pipeline_ace_step.py`):** The `__call__` method uses the `UMT5EncoderModel` for prompt embeddings and the `VoiceBpeTokenizer` for lyric tokenization.
3.  **Diffusion Process (`pipeline_ace_step.py`):**
    a. The `text2music_diffusion_process` method creates an initial random noise latent.
    b. It loops for `n` steps. In each step, it gets a prediction from `ACEStepTransformer2DModel.decode()` and uses a `Scheduler.step()` function to denoise the latent.
4.  **Audio Decoding (`music_dcae_pipeline.py`):** The final clean latent is passed to `MusicDCAE.decode()`.
5.  **Output:** The `MusicDCAE` uses its internal vocoder to convert the latent into a waveform, which is saved to a file.

---

## 3. File-by-File Technical Breakdown

### Root Directory

-   **`train_cli.py`**: A user-friendly CLI for launching LoRA training. It parses arguments, sets up logging, and orchestrates `trainer.py`.
-   **`trainer.py`**: Contains the core `pytorch_lightning.LightningModule` (`Pipeline`) for training. It handles model loading, LoRA injection, the training step, and optimizer configuration.
-   **`infer.py`**: A simple CLI for running a single inference test. It instantiates and calls the `ACEStepPipeline`.
-   **`infer-api.py`**: A FastAPI server that exposes the inference pipeline as a REST API at the `/generate` endpoint.
-   **`convert2hf_dataset.py`**: A data preprocessing script to convert raw audio/text files into the Hugging Face Datasets format needed for training.
-   **`setup.py`**: Standard Python setup script. It makes the project installable (`pip install .`) and creates the `acestep` command-line entry point.
-   **`Dockerfile` / `docker-compose.yaml`**: Define the containerized build and deployment environment for the application.

### `acestep/` - Core Package

-   **`pipeline_ace_step.py`**: The central orchestrator for all inference. The `ACEStepPipeline` class owns all model components and contains the main `__call__` method that chains together the full generation process.
-   **`gui.py`**: The entry point for the Gradio UI. It's registered as the `acestep` console script. It instantiates `ACEStepPipeline` and passes its `__call__` method to the UI components.
-   **`text2music_dataset.py`**: The PyTorch `Dataset` class for training. It loads data from disk and uses `language_segmentation` and `lyrics_utils` to process text and lyrics into tokens. Its `collate_fn` pads batches for the model.
-   **`apg_guidance.py`**: Implements different Classifier-Free Guidance (CFG) algorithms, such as Adaptive Prompt Guidance (`apg_forward`), which are used during the diffusion sampling loop.
-   **`cpu_offload.py`**: Provides the `@cpu_offload` decorator to automatically move models to/from the GPU to save VRAM during inference.
-   **`data_sampler.py`**: A utility to sample random generation parameters from JSON files in the `examples/` directory, used to populate the UI with examples.

### `acestep/models/` - Neural Network Architectures

-   **`ace_step_transformer.py`**: Defines the primary diffusion model, `ACEStepTransformer2DModel`. It is composed of `LinearTransformerBlock`s and handles all conditioning inputs (text, lyrics, speaker).
-   **`attention.py`**: Defines the `LinearTransformerBlock`, the main building block of the transformer. It combines self-attention, cross-attention, and a feed-forward network.
-   **`customer_attention_processor.py`**: Implements the low-level attention mechanisms. `CustomLiteLAProcessor2_0` provides an efficient linear attention implementation suitable for long audio sequences.
-   **`config.json`**: Stores the default hyperparameters (layer count, dimensions, heads, etc.) for the `ACEStepTransformer2DModel`.

### `acestep/models/lyrics_utils/` - Lyric Processing

-   **`lyric_encoder.py`**: Defines the `ConformerEncoder`, a powerful transformer-based model used to create high-quality embeddings from the lyric tokens.
-   **`lyric_tokenizer.py`**: Contains the `VoiceBpeTokenizer`, which handles multilingual text cleaning, normalization (via `multilingual_cleaners`), and tokenization using the `vocab.json`.
-   **`vocab.json`**: The vocabulary file mapping tokens (including special language and structure tokens) to integer IDs for the BPE tokenizer.
-   **`zh_num2words.py`**: A utility for converting numbers in Chinese text into their character-based representation, used during text normalization.

### `acestep/music_dcae/` - Audio Codec

-   **`music_dcae_pipeline.py`**: Defines the `MusicDCAE` class, a pipeline that wraps the autoencoder and vocoder to handle the `waveform <-> latent` conversion.
-   **`music_vocoder.py`**: Defines the `ADaMoSHiFiGANV1` model, which is a vocoder that generates audio waveforms from mel-spectrograms.
-   **`music_log_mel.py`**: Defines the `LogMelSpectrogram` module, used to convert raw audio into the mel-spectrograms that the vocoder and autoencoder operate on.

### `acestep/schedulers/` - Diffusion Schedulers

-   **`scheduling_*.py`**: Each file implements a different algorithm for the reverse diffusion process. They define the noise schedule and the `step` function that denoises the latent at each inference step. `FlowMatchEulerDiscreteScheduler` is a common choice.

### `acestep/ui/` - User Interface

-   **`components.py`**: Defines the entire Gradio UI layout and all its interactive components and event handlers. It receives the `ACEStepPipeline.__call__` function and wires it to the "Generate" button.
