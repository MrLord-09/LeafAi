
import argparse, os, tensorflow as tf
p=argparse.ArgumentParser(); p.add_argument('--model',default='backend/plant_disease_model.keras'); p.add_argument('--output',default='backend/model_quantized.tflite'); a=p.parse_args()
model=tf.keras.models.load_model(a.model); c=tf.lite.TFLiteConverter.from_keras_model(model); c.optimizations=[tf.lite.Optimize.DEFAULT]; data=c.convert(); os.makedirs(os.path.dirname(a.output),exist_ok=True); open(a.output,'wb').write(data); print('Saved',a.output)
