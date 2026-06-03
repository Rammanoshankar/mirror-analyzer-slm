#!/usr/bin/env python3
"""
Phase 6: SLM Inference
Runs Vision SLM (CNN) + Text SLM (Phi-3) inference pipeline
"""

import cv2
import numpy as np
import tensorflow as tf
import requests
import json
from pathlib import Path
import time
from datetime import datetime

class HealthAnalyzerSLM:
    def __init__(self):
        """
        Initialize SLM inference pipeline
        """
        print("Initializing Mirror Analyzer SLM...")
        
        # Load Vision SLM (TFLite)
        self.load_vision_model()
        
        # Load model info
        self.load_model_info()
        
        # Check Ollama connection
        self.check_ollama()
    
    def load_vision_model(self):
        """
        Load TFLite skin disease model
        """
        model_path = 'models/skin_disease_model.tflite'
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return False
        
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        print(f"✅ Vision SLM loaded: {model_path}")
        return True
    
    def load_model_info(self):
        """
        Load model metadata
        """
        info_path = 'models/model_info.json'
        
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.categories = self.model_info['categories']
        print(f"✅ Model categories loaded: {len(self.categories)} classes")
    
    def check_ollama(self):
        """
        Check if Ollama is running
        """
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is running at localhost:11434")
                return True
        except:
            print("⚠️  Ollama not running. Start with: ollama serve")
            return False
    
    def analyze_skin(self, image):
        """
        Analyze skin using Vision SLM
        """
        # Resize and normalize
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], img_input)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get predictions
        predictions = output_data[0]
        top_idx = np.argmax(predictions)
        top_confidence = predictions[top_idx]
        
        results = {
            'disease': self.categories[top_idx],
            'confidence': float(top_confidence),
            'all_predictions': {self.categories[i]: float(predictions[i]) for i in range(len(self.categories))}
        }
        
        return results
    
    def generate_health_report(self, skin_analysis, vital_signs=None):
        """
        Generate health report using Text SLM (Phi-3)
        """
        # Prepare prompt
        skin_text = f"""
Skin Analysis:
- Detected condition: {skin_analysis['disease']}
- Confidence: {skin_analysis['confidence']*100:.1f}%
"""
        
        if vital_signs:
            vital_text = f"""
Vital Signs:
- Body Temperature: {vital_signs.get('temperature', 'N/A')}°C
- Heart Rate: {vital_signs.get('heart_rate', 'N/A')} BPM
- Blood Oxygen (SpO2): {vital_signs.get('spo2', 'N/A')}%
"""
        else:
            vital_text = ""
        
        prompt = f"""{skin_text}{vital_text}
Based on the above analysis, provide a brief health assessment and recommendations."""
        
        # Call Ollama
        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'mirror-analyzer-slm',
                    'prompt': prompt,
                    'stream': False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return "Could not generate report"
        except Exception as e:
            return f"Error generating report: {str(e)}"
    
    def process_webcam_frame(self):
        """
        Process webcam frame and generate analysis
        """
        cap = cv2.VideoCapture(0)
        
        print("\n" + "="*70)
        print("Mirror Analyzer SLM - Real-time Inference")
        print("="*70)
        print("\nPress 'c' to capture and analyze")
        print("Press 'q' to quit\n")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Display frame
            cv2.imshow('Mirror Analyzer - Press C to Capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Analyze
                print("\nAnalyzing...")
                start_time = time.time()
                
                skin_analysis = self.analyze_skin(frame)
                inference_time = (time.time() - start_time) * 1000
                
                print(f"\n{'='*70}")
                print("Analysis Results")
                print(f"{'='*70}")
                print(f"\nSkin Disease Detected: {skin_analysis['disease']}")
                print(f"Confidence: {skin_analysis['confidence']*100:.1f}%")
                print(f"Vision SLM Inference Time: {inference_time:.2f} ms")
                
                print(f"\nAll Predictions:")
                for disease, confidence in sorted(
                    skin_analysis['all_predictions'].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    bar_length = int(confidence * 20)
                    bar = '█' * bar_length
                    print(f"  {disease:<15} {bar:<20} {confidence*100:>6.2f}%")
                
                print(f"\nGenerating health report...")
                report = self.generate_health_report(skin_analysis)
                
                print(f"\n{'='*70}")
                print("Health Report")
                print(f"{'='*70}")
                print(report)
                print(f"{'='*70}\n")
                
                # Save results
                self.save_analysis(skin_analysis, report, frame)
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_analysis(self, skin_analysis, report, frame):
        """
        Save analysis results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        image_path = f'results/analysis_{timestamp}.jpg'
        cv2.imwrite(image_path, frame)
        
        # Save results
        results = {
            'timestamp': timestamp,
            'skin_analysis': skin_analysis,
            'report': report
        }
        
        results_path = f'results/analysis_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved: {results_path}")

if __name__ == '__main__':
    # Create results folder
    Path('results').mkdir(exist_ok=True)
    
    # Initialize
    analyzer = HealthAnalyzerSLM()
    
    # Run
    analyzer.process_webcam_frame()
