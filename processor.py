import cv2
import numpy as np
import json
import os
import logging
import asyncio
import websockets
from threading import Thread, Lock
from queue import Queue
from collections import deque
import onnxruntime as rt
from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processor.log"),
        logging.StreamHandler()
    ]
)

class SignLanguageProcessor:
    def __init__(self):
        self.config = self.load_configs()
        self.models = {}
        self.labels = {}
        self.current_model_type = 'RSL'
        self.buffer = None
        self.lock = Lock()
        self.init_models()
        self.init_buffers()

    def load_configs(self):
        """Загрузка конфигураций для обеих моделей"""
        configs = {}
        try:
            with open('config.json') as f:
                configs['RSL'] = json.load(f)
            with open('lastconfig.json') as f:
                configs['ASL'] = json.load(f)
            
            # Валидация обязательных полей
            required_keys = {
                'RSL': ['path_to_model', 'threshold', 'topk', 
                       'path_to_class_list', 'window_size', 'provider'],
                'ASL': ['path_to_model', 'threshold', 'topk',
                       'path_to_class_list', 'window_size']
            }
            
            for model_type in ['RSL', 'ASL']:
                for key in required_keys[model_type]:
                    if key not in configs[model_type]:
                        raise KeyError(f"Missing key '{key}' in {model_type} config")
            
            return configs
        except Exception as e:
            logging.error(f"Config load error: {str(e)}")
            raise

    def init_models(self):
        """Инициализация моделей с учетом конфигов"""
        try:
            # Инициализация модели RSL
            self.models['RSL'] = rt.InferenceSession(
                self.config['RSL']['path_to_model'],
                providers=[self.config['RSL']['provider']]
            )
            
            # Инициализация модели ASL
            self.models['ASL'] = load_model(
                self.config['ASL']['path_to_model']
            )
            
            # Загрузка меток для текущей модели
            self.load_labels()
            logging.info("Models initialized successfully")

        except Exception as e:
            logging.error(f"Model init error: {str(e)}")
            raise

    def init_buffers(self):
        """Инициализация буферов согласно конфигурации"""
        with self.lock:
            self.buffer = deque(
                maxlen=self.config[self.current_model_type]['window_size']
            )

    def load_labels(self):
        """Загрузка меток классов для активной модели"""
        label_path = self.config[self.current_model_type]['path_to_class_list']
        self.labels = {}
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t' if self.current_model_type == 'RSL' else ' ')
                    if len(parts) < 2:
                        logging.warning(f"Invalid format in line {line_num}")
                        continue
                    self.labels[int(parts[0])] = parts[1]
            logging.info(f"Loaded {len(self.labels)} labels for {self.current_model_type}")
        except Exception as e:
            logging.error(f"Label load error: {str(e)}")
            self.labels = {}

    async def process_frame(self, frame_data):
        """Обработка кадра с учетом текущей конфигурации"""
        try:
            frame = cv2.imdecode(
                np.frombuffer(frame_data, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            # Предобработка кадра
            processed = self.preprocess_frame(frame)
            
            # Добавление в буфер
            with self.lock:
                self.buffer.append(processed)
                
                if len(self.buffer) >= self.config[self.current_model_type]['window_size']:
                    return self.predict_and_format()
                
            return {'gestures': []}
            
        except Exception as e:
            logging.error(f"Frame processing error: {str(e)}")
            return {'error': str(e)}

    def preprocess_frame(self, frame):
        """Предобработка кадра для текущей модели"""
        if self.current_model_type == 'RSL':
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame.astype(np.float32) / 255.0
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (28, 28))
            return np.expand_dims(frame, axis=-1).astype(np.float32) / 255.0

    def predict_and_format(self):
        """Выполнение предсказания и форматирование результатов"""
        conf = self.config[self.current_model_type]
        
        if self.current_model_type == 'RSL':
            clip = np.transpose(np.array(self.buffer), (3, 0, 1, 2))
            inputs = {self.models['RSL'].get_inputs()[0].name: np.expand_dims(clip, 0)}
            outputs = self.models['RSL'].run(None, inputs)[0][0]
        else:
            outputs = self.models['ASL'].predict(np.array(self.buffer), verbose=0)[0]
        
        probs = self.softmax(outputs)
        top_indices = np.argsort(probs)[-conf['topk']:][::-1]
        
        results = []
        for idx in top_indices:
            if probs[idx] > conf['threshold']:
                results.append({
                    'label': self.labels.get(idx, "Unknown"),
                    'confidence': float(probs[idx])
                })
        
        return {'gestures': results[:conf['topk']}

    def softmax(self, x):
        """Оптимизированная функция softmax"""
        max_x = np.max(x)
        e_x = np.exp(x - max_x)
        return e_x / e_x.sum()

    async def websocket_handler(self, websocket, path):
        """Обработчик WebSocket соединений"""
        try:
            async for message in websocket:
                if message.startswith('switch_model:'):
                    new_model = message.split(':')[1]
                    await self.switch_model(new_model, websocket)
                else:
                    result = await self.process_frame(message)
                    await websocket.send(json.dumps(result))
        except Exception as e:
            logging.error(f"WS error: {str(e)}")

    async def switch_model(self, new_model, websocket):
        """Переключение между моделями"""
        with self.lock:
            if new_model in ['RSL', 'ASL'] and new_model != self.current_model_type:
                self.current_model_type = new_model
                self.init_buffers()
                self.load_labels()
                logging.info(f"Switched to {new_model} model")
                await websocket.send(json.dumps({
                    'status': 'model_changed',
                    'new_model': new_model
                }))

def start_processor():
    processor = SignLanguageProcessor()
    
    async def run_server():
        async with websockets.serve(
            processor.websocket_handler,
            "0.0.0.0",
            8765
        ):
            await asyncio.Future()
    
    def run_in_thread():
        asyncio.run(run_server())
    
    Thread(target=run_in_thread, daemon=True).start()
    logging.info("Processor started on port 8765")
    return processor

if __name__ == "__main__":
    try:
        processor = start_processor()
        while True:
            pass
    except KeyboardInterrupt:
        logging.info("Processor stopped")
