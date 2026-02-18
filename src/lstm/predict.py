import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.lstm.tokenizer import SentencePieceTokenizer
from src.lstm.model import BiLSTMSpamClassifier


def predict_spam(text):
    root = Path(__file__).parent.parent.parent
    checkpoint = torch.load(root / 'data' / 'models' / 'best_model.pt', map_location='cpu')
    info = checkpoint['model_info']
    
    tokenizer = SentencePieceTokenizer(str(root / 'data' / 'models' / 'lstm_tokenizer.model'))
    model = BiLSTMSpamClassifier(**{k: info[k] for k in ['vocab_size', 'embedding_dim',     'hidden_size', 
                                                         'num_layers', 'dropout_rate', 'dense_hidden']},
                                  padding_idx=tokenizer.pad_id)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    ids = tokenizer.encode(text)[:512] + [tokenizer.pad_id] * (512 - len(tokenizer.encode(text)[:512]))
    mask = [1] * len(tokenizer.encode(text)[:512]) + [0] * (512 - len(tokenizer.encode(text)[:512]))
    
    with torch.no_grad():
        prob = model(torch.tensor([ids]), torch.tensor([mask])).item()
    
    return 'spam' if prob > 0.5 else 'ham', prob


if __name__ == '__main__':
    text = input("Enter text: ") if len(sys.argv) == 1 else ' '.join(sys.argv[1:])
    pred, prob = predict_spam(text)
    print(f"{pred.upper()} ({prob:.2%})")