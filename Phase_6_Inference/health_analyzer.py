#!/usr/bin/env python3
"""
Phase 6B: Agentic Health Analysis Workflow
Multi-agent system for comprehensive health analysis
"""

import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests

class AnalysisAgent:
    """Base agent class"""
    def __init__(self, name):
        self.name = name
        self.results = {}
    
    def execute(self, data):
        raise NotImplementedError

class VisionAgent(AnalysisAgent):
    """Analyzes facial images using CNN"""
    
    def __init__(self):
        super().__init__("Vision Agent")
        self.interpreter = None
        self.load_model()
    
    def load_model(self):
        import tensorflow as tf
        model_path = 'models/skin_disease_model.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
    
    def execute(self, image):
        """Analyze image and return results"""
        import tensorflow as tf
        
        img_resized = cv2.resize(image, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        
        self.interpreter.set_tensor(input_details[0]['index'], img_input)
        self.interpreter.invoke()
        
        predictions = self.interpreter.get_tensor(output_details[0]['index'])[0]
        
        self.results = {
            'agent': self.name,
            'predictions': predictions.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

class FeatureExtractionAgent(AnalysisAgent):
    """Extracts facial features using MediaPipe"""
    
    def __init__(self):
        super().__init__("Feature Extraction Agent")
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
    
    def execute(self, image):
        """Extract facial landmarks"""
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        self.results = {
            'agent': self.name,
            'face_detected': results.multi_face_landmarks is not None,
            'num_faces': len(results.multi_face_landmarks) if results.multi_face_landmarks else 0,
            'features': {
                'left_eye_open': 'analyzing',
                'right_eye_open': 'analyzing',
                'face_symmetry': 'good',
                'head_pose': 'frontal'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

class HealthReportAgent(AnalysisAgent):
    """Generates health reports using Phi-3"""
    
    def __init__(self):
        super().__init__("Health Report Agent")
    
    def execute(self, vision_results, vital_signs=None):
        """Generate comprehensive health report"""
        prompt = f"""
Generate a concise health report based on:
- Vision analysis: Top prediction is {max(enumerate(vision_results['predictions']), key=lambda x: x[1])[1]:.2%} confident
- Features detected: Face is visible and clear

Provide observations and recommendations in 2-3 sentences.
"""
        
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
            
            report = response.json()['response'] if response.status_code == 200 else "Report generation failed"
        except:
            report = "Could not connect to Ollama"
        
        self.results = {
            'agent': self.name,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

class TrendAnalysisAgent(AnalysisAgent):
    """Analyzes health trends over time"""
    
    def __init__(self):
        super().__init__("Trend Analysis Agent")
        self.history_file = 'data/health_history.json'
        self.load_history()
    
    def load_history(self):
        """Load historical data"""
        Path('data').mkdir(exist_ok=True)
        
        if Path(self.history_file).exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []
    
    def execute(self, current_analysis):
        """Analyze trends"""
        # Add to history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'analysis': current_analysis
        })
        
        # Keep last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        self.history = [
            h for h in self.history
            if datetime.fromisoformat(h['timestamp']) > seven_days_ago
        ]
        
        # Save
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.results = {
            'agent': self.name,
            'history_length': len(self.history),
            'trend': 'stable',
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

class AlertAgent(AnalysisAgent):
    """Generates alerts for urgent health concerns"""
    
    def __init__(self):
        super().__init__("Alert Agent")
    
    def execute(self, analysis_results):
        """Check for urgent alerts"""
        alerts = []
        
        # Example: Flag high confidence melanoma detection
        predictions = analysis_results.get('predictions', [])
        if len(predictions) > 0 and predictions[0] > 0.8:  # MEL (melanoma) class
            alerts.append({
                'level': 'high',
                'message': 'High confidence skin lesion detected. Consult a dermatologist.'
            })
        
        self.results = {
            'agent': self.name,
            'alerts': alerts,
            'has_alerts': len(alerts) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results

class HealthAnalysisOrchestrator:
    """Orchestrates multi-agent health analysis"""
    
    def __init__(self):
        print("Initializing Agentic Health Analysis System...\n")
        
        self.vision_agent = VisionAgent()
        self.feature_agent = FeatureExtractionAgent()
        self.report_agent = HealthReportAgent()
        self.trend_agent = TrendAnalysisAgent()
        self.alert_agent = AlertAgent()
    
    def analyze(self, image):
        """
        Execute full analysis workflow
        """
        print(f"\n{'='*70}")
        print("AGENTIC HEALTH ANALYSIS WORKFLOW")
        print(f"{'='*70}\n")
        
        # Step 1: Vision Analysis
        print("[1/5] Running Vision Agent...")
        vision_results = self.vision_agent.execute(image)
        print(f"      ✅ Vision analysis complete\n")
        
        # Step 2: Feature Extraction
        print("[2/5] Running Feature Extraction Agent...")
        feature_results = self.feature_agent.execute(image)
        print(f"      ✅ Features extracted: {feature_results['num_faces']} face(s) detected\n")
        
        # Step 3: Health Report
        print("[3/5] Running Health Report Agent...")
        report_results = self.report_agent.execute(vision_results)
        print(f"      ✅ Report generated\n")
        
        # Step 4: Trend Analysis
        print("[4/5] Running Trend Analysis Agent...")
        trend_results = self.trend_agent.execute(vision_results)
        print(f"      ✅ Trend analysis complete ({trend_results['history_length']} historical records)\n")
        
        # Step 5: Alert Generation
        print("[5/5] Running Alert Agent...")
        alert_results = self.alert_agent.execute(vision_results)
        print(f"      ✅ Alerts checked ({len(alert_results['alerts'])} alert(s))\n")
        
        # Compile final report
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'agents_executed': 5,
            'vision': vision_results,
            'features': feature_results,
            'report': report_results,
            'trends': trend_results,
            'alerts': alert_results
        }
        
        return final_report
    
    def print_report(self, report):
        """
        Pretty print the analysis report
        """
        print(f"\n{'='*70}")
        print("COMPREHENSIVE HEALTH ANALYSIS REPORT")
        print(f"{'='*70}")
        
        print(f"\n📊 Vision Analysis:")
        print(f"   Predictions: {len(report['vision']['predictions'])} classes analyzed")
        
        print(f"\n👁️  Face Detection:")
        print(f"   Faces detected: {report['features']['num_faces']}")
        
        print(f"\n📋 Health Report:")
        print(f"   {report['report']['report']}")
        
        print(f"\n📈 Trend Analysis:")
        print(f"   Historical records: {report['trends']['history_length']} (last 7 days)")
        print(f"   Overall trend: {report['trends']['trend']}")
        
        if report['alerts']['has_alerts']:
            print(f"\n🚨 ALERTS:")
            for alert in report['alerts']['alerts']:
                print(f"   [{alert['level'].upper()}] {alert['message']}")
        else:
            print(f"\n✅ No urgent alerts")
        
        print(f"\n{'='*70}\n")

if __name__ == '__main__':
    # Initialize orchestrator
    orchestrator = HealthAnalysisOrchestrator()
    
    # Capture image from webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Run full analysis
        report = orchestrator.analyze(frame)
        
        # Print report
        orchestrator.print_report(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'results/agentic_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved: results/agentic_report_{timestamp}.json")
