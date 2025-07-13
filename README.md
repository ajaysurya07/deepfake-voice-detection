 Deepfake Voice Detection Project

 Overview
This project focuses on detecting deepfake audio (voice spoofing) using a deep learning approach. The model leverages Wav2Vec 2.0 features combined with advanced signal processing techniques to distinguish between genuine and synthetic voices.

 Key Features
- Advanced Architecture: Combines Wav2Vec 2.0 with multi-scale feature analysis
- Robust Training: Uses focal loss and label smoothing for class imbalance
- Data Augmentation: Applies various audio transformations to improve generalization
- Performance Metrics: Achieves ~97% validation accuracy on ASVspoof 2019 dataset

 Dataset
The model is trained on the ASVspoof 2019 LA dataset:
- Training set: 25,380 samples (89.8% spoof, 10.2% bonafide)
- Evaluation set: 49,865 samples

 Model Architecture
The AntiSpoofModel includes:
1. Wav2Vec 2.0 backbone (frozen feature extractor)
2. Multi-scale F0 analysis (low/mid/high frequency bands)
3. Spectral inconsistency detector
4. Multi-branch CNN (short/medium/long temporal scales)
5. Attention mechanisms (temporal and channel attention)
6. Transformer encoder for sequence modeling
7. Advanced pooling (average + max pooling)

 Training Details
- Loss Function: Label smoothing loss (smoothing=0.1)
- Optimizer: AdamW with differential learning rates
- LR Scheduling: Cosine annealing with warm restarts
- Batch Size: 8
- Epochs: 5 (early stopping monitored)

 Performance
- Training Accuracy: 97.9%
- Validation Accuracy: 97.1%
- Test Accuracy: [To be evaluated on your specific test set]

 Usage

 Installation
bash
pip install torch torchaudio transformers pandas tqdm audiomentations


 Loading the Model
python
model = AntiSpoofModel(
    wav2vec_model_path="path/to/wav2vec",
    use_transformer=True,
    num_classes=2
).to(device)
model.load_state_dict(torch.load("antispoof_model.pth"))


 Making Predictions
python
def predict_audio(path):
    waveform = preprocess_audio(path).to(device)
    with torch.no_grad():
        output = model(waveform)
        probs = F.softmax(output['logits'], dim=1)
        pred = torch.argmax(probs).item()
    return "Real" if pred == 0 else "Fake", probs[0][pred].item()


 File Structure

deepfake-voice-detection/
├── deepfake-voice-detection.ipynb    Main training notebook
├── best_improved_antispoofmodel.pth  Trained model weights
├── README.md                         This file
└── requirements.txt                  Python dependencies


 Future Improvements
- Add real-time audio processing capability
- Implement ensemble methods for improved robustness
- Extend to other spoofing attack types (A07-A19)
- Deploy as a web service with audio upload interface

 Acknowledgments
- ASVspoof 2019 dataset
- Hugging Face Transformers library
- PyTorch audio processing utilities
