import librosa as lr

def load_wav(audio_path, scaling=1, delta=1, chroma=1, mel=1):
    
      return lr.load(audio_path)
