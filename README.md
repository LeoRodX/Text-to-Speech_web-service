# Text-to-Speech_web-service
### Краткая инструкция по развертыванию веб-приложения TTS (Text-to-Speech)

#### 1. Установка зависимостей
```bash
sudo apt update
sudo apt install -y python3 python3-venv wget
```

#### 2. Создание и активация виртуальной среды
```bash
mkdir -p ~/tts-4sp && cd ~/tts-4sp
python3 -m venv venv && source venv/bin/activate
```

#### 3. Установка Python-зависимостей
```bash
pip install torch flask soundfile scipy
```

#### 4. Загрузка модели Silero TTS
```bash
wget https://models.silero.ai/models/tts/ru/v4_ru.pt -O silero_v4_ru.pt
```

#### 5. Создание файла приложения
Скопируйте содержимое `app.py` в файл `~/tts-4sp/app.py`:
```bash
nano ~/tts-4sp/app.py
# Вставьте код и сохраните, Ctrl+X
```

#### 6. Запуск приложения
```bash
source venv/bin/activate  # Активируйте среду, если ещё не активирована
python app.py
```

#### 7. Доступ к приложению
Откройте в браузере:  
http://localhost:5000  
или  
http://[IP-адрес-сервера]:5000

#### Логи:
- Аудиофайлы сохраняются в `~/tts-4sp/audio_output/`
- Ошибки выводятся в консоль и в веб-интерфейс

Приложение готово к использованию! Для остановки нажмите `Ctrl+C` в терминале.
