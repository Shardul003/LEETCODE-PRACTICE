from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('model_max_length =', getattr(t, 'model_max_length', None))
print('derived chunk_size =', max(128, (getattr(t, 'model_max_length', None) or 512) - 10))
print('suggested overlap =', min(128, max(0, (max(128, (getattr(t, "model_max_length", None) or 512) - 10)) // 10)))
