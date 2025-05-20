import tensorflow as tf
import os

class ModelConverter:
    """Convert and optimize models for Raspberry Pi"""
    
    @staticmethod
    def convert_to_tflite(model, output_path):
        """Convert TF model to TFLite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
    @staticmethod
    def quantize_model(model, output_path, dataset_gen):
        """Quantize model to reduce size and improve inference speed"""
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Set optimization flags
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Representative dataset for quantization
        def representative_dataset():
            for _ in range(100):
                data = next(dataset_gen)[0]
                yield [data]
                
        converter.representative_dataset = representative_dataset
        
        # Convert and save
        tflite_model = converter.convert()
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
    @staticmethod
    def optimize_for_edge(model, output_path, dataset_gen):
        """Full optimization pipeline for edge deployment"""
        # 1. Convert to TFLite
        tflite_path = os.path.join(output_path, 'model.tflite')
        ModelConverter.convert_to_tflite(model, tflite_path)
        
        # 2. Quantize model
        quantized_path = os.path.join(output_path, 'model_quantized.tflite')
        ModelConverter.quantize_model(model, quantized_path, dataset_gen)
        
        # 3. Print model info
        interpreter = tf.lite.Interpreter(model_path=quantized_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print("\nOptimized Model Details:")
        print("Input Shape:", input_details[0]['shape'])
        print("Output Shape:", output_details[0]['shape'])
        print("Original Size:", os.path.getsize(tflite_path) / 1024 / 1024, "MB")
        print("Quantized Size:", os.path.getsize(quantized_path) / 1024 / 1024, "MB")
        
        return quantized_path
