import os
import time
import torch
import numpy as np
from scipy.signal import lfilter, butter
import soundfile as sf
from flask import Flask, request, send_file, render_template_string

# Инициализация Flask
app = Flask(__name__)

# Конфигурация
TTS_MODEL_PATH = os.path.expanduser('~/tts-4sp/silero_v4_ru.pt')
AUDIO_DIR = os.path.expanduser('~/tts-4sp/audio_output')
os.makedirs(AUDIO_DIR, exist_ok=True)

# Оптимизация для Intel N100
torch.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

# Голоса и их параметры
VOICES = {
    'xenia': {'name': 'Ксения (женский)', 'gender': 'female', 'eq': False},
    'aidar': {'name': 'Айдар (мужской)', 'gender': 'male', 'eq': True},
    'baya': {'name': 'Бая (женский альт.)', 'gender': 'female', 'eq': False},
    'eugene': {'name': 'Евгений (мужской альт.)', 'gender': 'male', 'eq': True}
}

# Загрузка модели
def load_model():
    global model
    if not os.path.exists(TTS_MODEL_PATH):
        raise FileNotFoundError(f"Файл модели не найден: {TTS_MODEL_PATH}")
    
    try:
        model = torch.package.PackageImporter(TTS_MODEL_PATH).load_pickle("tts_models", "model")
        model.to(torch.device('cpu'))
        print("✓ Модель загружена. Доступные голоса:", list(VOICES.keys()))
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели: {str(e)}")

# Обработка аудио
def apply_audio_effects(audio, sample_rate, voice_params):
    """Применение эффектов к аудио"""
    audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
    
    # 1. Нормализация
    audio_np = 0.9 * audio_np / np.max(np.abs(audio_np))
    
    # 2. Коррекция тембра (для мужских голосов)
    if voice_params['eq']:
        b, a = butter(4, [100, 3000], 'bandpass', fs=sample_rate)
        audio_np = lfilter(b, a, audio_np)
    
    # 3. Реверберация
    impulse = np.zeros(int(0.1 * sample_rate))
    # impulse[0] = 1.0
    impulse[0] = 0.0
    # impulse[int(0.05 * sample_rate)] = 0.6
    impulse[int(0.05 * sample_rate)] = 0.0
    
    reverb = np.convolve(audio_np, impulse, mode='same')
    processed_audio = 0.85 * audio_np + 0.15 * reverb
    
    return processed_audio

# Генерация речи
def generate_speech(text, speaker='xenia', sample_rate=48000):
    if speaker not in VOICES:
        raise ValueError(f"Неизвестный голос: {speaker}")
    
    start_time = time.time()
    
    # Генерация исходного аудио
    audio = model.apply_tts(
        text=text,
        speaker=speaker,
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True
    )
    
    # Обработка эффектов
    processed_audio = apply_audio_effects(
        audio,
        sample_rate,
        VOICES[speaker]
    )
    
    # Сохранение
    filename = f"{speaker}_{int(time.time())}.wav"
    output_path = os.path.join(AUDIO_DIR, filename)
    sf.write(output_path, processed_audio, samplerate=sample_rate)
    
    print(f"✓ Сгенерировано: {filename} ({len(text)} символов, {time.time()-start_time:.2f} сек)")
    return output_path, filename

# Веб-интерфейс
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Генератор русской речи</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; text-align: center; }
        textarea { width: 100%; height: 120px; padding: 10px; margin: 10px 0; border: 1px solid #ddd; }
        select, button { padding: 10px 15px; font-size: 16px; margin: 5px 0; }
        button { background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .audio-container { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }
        .error { color: red; padding: 10px; background: #ffeeee; border-radius: 5px; }
        .download-link { display: inline-block; margin-top: 10px; color: #06c; }
    </style>
</head>
<body>
    <h1>Генератор русской речи</h1>
    
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
    
    <form method="post">
        <textarea name="text" placeholder="Введите текст на русском..." required>{{ text if text }}</textarea><br>
        
        <select name="voice">
            {% for id, params in voices.items() %}
                <option value="{{ id }}" {{ 'selected' if voice == id }}>
                    {{ params.name }}
                </option>
            {% endfor %}
        </select>
        
        <button type="submit">Сгенерировать речь</button>
    </form>
    
    {% if audio_file %}
    <div class="audio-container">
        <audio controls autoplay style="width: 100%">
            <source src="{{ url_for('serve_audio', filename=audio_file) }}" type="audio/wav">
            Ваш браузер не поддерживает аудио элементы.
        </audio>
        <a class="download-link" href="{{ url_for('serve_audio', filename=audio_file) }}" download>
            Скачать WAV-файл
        </a>
    </div>
    {% endif %}
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        speaker = request.form.get('voice', 'xenia')
        
        if not text:
            return render_template_string(HTML_TEMPLATE,
                                       voices=VOICES,
                                       error="Введите текст для генерации!")
        
        if len(text) > 1000:
            return render_template_string(HTML_TEMPLATE,
                                       voices=VOICES,
                                       text=text,
                                       error="Максимум 1000 символов!")
        
        try:
            _, filename = generate_speech(text, speaker)
            return render_template_string(HTML_TEMPLATE,
                                       voices=VOICES,
                                       text=text,
                                       voice=speaker,
                                       audio_file=filename)
        except Exception as e:
            return render_template_string(HTML_TEMPLATE,
                                       voices=VOICES,
                                       text=text,
                                       error=f"Ошибка генерации: {str(e)}")
    
    return render_template_string(HTML_TEMPLATE, voices=VOICES)

@app.route('/audio/<filename>')
def serve_audio(filename):
    path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(path):
        return "Аудиофайл не найден", 404
    return send_file(path, mimetype='audio/wav')

# Инициализация
if __name__ == '__main__':
    try:
        load_model()
        print("Сервер запущен: http://localhost:5000")
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Ошибка запуска: {str(e)}")
        exit(1)
