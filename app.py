"""
KrishiMitra AI - AI Models and NLP Processing
Advanced AI components for multilingual agricultural advisory
Team: The Innovators
"""

import re
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

class IndicBERTProcessor:
    """
    IndicBERT-based multilingual text processing for Indian languages
    Handles Hindi, English, and regional language queries
    """
    
    def __init__(self):
        self.supported_languages = ['hi', 'en', 'pa', 'bn', 'te', 'mr', 'gu', 'ta']
        self.intent_patterns = {
            'weather': [
                r'barish|rain|mausam|weather|paani|water|baarish',
                r'humidity|temperature|wind|climate'
            ],
            'irrigation': [
                r'sinchai|irrigation|paani dena|watering',
                r'kab paani de|when to water|irrigation timing'
            ],
            'market': [
                r'mandi|price|rate|bhav|market|sell|bechna',
                r'commodity price|market rate|selling price'
            ],
            'fertilizer': [
                r'khad|fertilizer|urvarak|nutrients|manure',
                r'NPK|urea|phosphate|potash|organic'
            ],
            'pest': [
                r'keeda|pest|insect|disease|bimari|fungus',
                r'crop disease|plant protection|pesticide'
            ],
            'scheme': [
                r'yojana|scheme|subsidy|government|sarkar',
                r'loan|credit|insurance|financial help'
            ],
            'crop': [
                r'fasal|crop|bija|seed|planting|cultivation',
                r'sowing|harvesting|crop calendar|variety'
            ]
        }
        
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Simple language detection based on script
        hindi_chars = re.findall(r'[\u0900-\u097F]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        if len(hindi_chars) > len(english_chars):
            return 'hi'
        elif len(english_chars) > 0:
            return 'en'
        else:
            return 'hi'  # Default to Hindi
            
    def extract_intent(self, text: str) -> Tuple[str, float]:
        """Extract intent from user query with confidence score"""
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches * 0.3
            intent_scores[intent] = score
            
        if not intent_scores or max(intent_scores.values()) == 0:
            return 'general', 0.5
            
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[best_intent], 1.0)
        
        return best_intent, confidence
        
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities like crop, location, quantity from text"""
        entities = {}
        
        # Crop detection
        crops = {
            'wheat': r'gehun|wheat|गेहूं',
            'rice': r'chawal|rice|धान|चावल',
            'cotton': r'kapas|cotton|कपास',
            'sugarcane': r'ganna|sugarcane|गन्ना',
            'potato': r'aloo|potato|आलू',
            'tomato': r'tamatar|tomato|टमाटर',
            'onion': r'pyaz|onion|प्याज'
        }
        
        for crop, pattern in crops.items():
            if re.search(pattern, text.lower()):
                entities['crop'] = crop
                break
                
        # Location detection
        locations = {
            'delhi': r'delhi|दिल्ली',
            'punjab': r'punjab|पंजाब',
            'haryana': r'haryana|हरियाणा',
            'up': r'uttar pradesh|up|उत्तर प्रदेश',
            'bihar': r'bihar|बिहार',
            'maharashtra': r'maharashtra|महाराष्ट्र'
        }
        
        for location, pattern in locations.items():
            if re.search(pattern, text.lower()):
                entities['location'] = location
                break
                
        # Quantity detection
        quantity_match = re.search(r'(\d+)\s*(kg|quintal|ton|acre|hectare)', text.lower())
        if quantity_match:
            entities['quantity'] = quantity_match.group(1)
            entities['unit'] = quantity_match.group(2)
            
        return entities

class RAGEngine:
    """
    Retrieval-Augmented Generation Engine for KrishiMitra AI
    Combines knowledge base with real-time data for accurate responses
    """
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.context_window = 5  # Number of previous interactions to consider
        
    def _load_knowledge_base(self) -> Dict:
        """Load agricultural knowledge base"""
        return {
            'crops': {
                'wheat': {
                    'sowing_season': 'Rabi (October-December)',
                    'harvesting': 'April-May',
                    'water_requirement': 'Medium (4-6 irrigations)',
                    'fertilizer': 'NPK 120:60:40 kg/hectare',
                    'varieties': ['HD-2967', 'PBW-343', 'DBW-17'],
                    'diseases': ['Rust', 'Bunt', 'Leaf blight'],
                    'ideal_temp': '15-25°C',
                    'soil_ph': '6.0-7.5'
                },
                'rice': {
                    'sowing_season': 'Kharif (May-July)',
                    'harvesting': 'October-December',
                    'water_requirement': 'High (standing water)',
                    'fertilizer': 'NPK 100:50:50 kg/hectare',
                    'varieties': ['Pusa-44', 'IR-64', 'Swarna'],
                    'diseases': ['Blast', 'Sheath blight', 'Brown spot'],
                    'ideal_temp': '20-35°C',
                    'soil_ph': '5.5-7.0'
                }
            },
            'weather_guidelines': {
                'irrigation': {
                    'high_humidity': 'Delay irrigation if humidity >80%',
                    'rainfall_expected': 'Skip irrigation if rain expected within 24 hours',
                    'temperature': 'Best irrigation time: early morning or evening'
                },
                'spraying': {
                    'wind_speed': 'Avoid spraying if wind speed >10 km/h',
                    'temperature': 'Spray when temperature <30°C',
                    'humidity': 'Best humidity range: 60-80%'
                }
            },
            'market_insights': {
                'price_factors': [
                    'Seasonal demand',
                    'Weather conditions',
                    'Government procurement',
                    'Export policies',
                    'Storage capacity'
                ],
                'selling_tips': [
                    'Monitor MSP announcements',
                    'Check multiple mandis',
                    'Consider storage costs',
                    'Track festival seasons'
                ]
            }
        }
        
    def retrieve_context(self, intent: str, entities: Dict, query: str) -> Dict:
        """Retrieve relevant context from knowledge base"""
        context = {'sources': [], 'facts': [], 'recommendations': []}
        
        # Crop-specific information
        if 'crop' in entities:
            crop = entities['crop']
            if crop in self.knowledge_base['crops']:
                crop_info = self.knowledge_base['crops'][crop]
                context['facts'].extend([
                    f"Sowing season: {crop_info['sowing_season']}",
                    f"Water requirement: {crop_info['water_requirement']}",
                    f"Recommended fertilizer: {crop_info['fertilizer']}"
                ])
                context['sources'].append(f"Crop Database - {crop.title()}")
                
        # Intent-specific guidelines
        if intent == 'irrigation':
            weather_guide = self.knowledge_base['weather_guidelines']['irrigation']
            context['recommendations'].extend(list(weather_guide.values()))
            context['sources'].append("Irrigation Best Practices")
            
        elif intent == 'market':
            market_info = self.knowledge_base['market_insights']
            context['facts'].extend(market_info['selling_tips'])
            context['sources'].append("Market Intelligence")
            
        return context
        
    def generate_response(self, intent: str, entities: Dict, context: Dict, 
                         real_time_data: Dict) -> Dict:
        """Generate comprehensive response using RAG approach"""
        
        response = {
            'primary_advice': '',
            'detailed_explanation': '',
            'action_items': [],
            'confidence_score': 0.85,
            'sources': context['sources'],
            'warnings': []
        }
        
        # Generate intent-specific responses
        if intent == 'weather' or intent == 'irrigation':
            response.update(self._generate_weather_response(
                entities, context, real_time_data))
                
        elif intent == 'market':
            response.update(self._generate_market_response(
                entities, context, real_time_data))
                
        elif intent == 'fertilizer':
            response.update(self._generate_fertilizer_response(
                entities, context, real_time_data))
                
        elif intent == 'pest':
            response.update(self._generate_pest_response(
                entities, context, real_time_data))
                
        elif intent == 'scheme':
            response.update(self._generate_scheme_response(
                entities, context, real_time_data))
                
        else:
            response.update(self._generate_general_response(
                entities, context, real_time_data))
        
        return response
        
    def _generate_weather_response(self, entities: Dict, context: Dict, 
                                 real_time_data: Dict) -> Dict:
        """Generate weather-based agricultural advice"""
        weather = real_time_data.get('weather', {}).get('current', {})
        
        temp = weather.get('temperature', 25)
        humidity = weather.get('humidity', 60)
        
        advice = f"Current conditions: {temp}°C, {humidity}% humidity. "
        
        if humidity > 80:
            advice += "High humidity detected. Delay irrigation and avoid fungicide spray."
            actions = ["Skip irrigation today", "Monitor for fungal diseases"]
        elif humidity < 40:
            advice += "Low humidity. Increase irrigation frequency."
            actions = ["Provide extra watering", "Mulch around plants"]
        else:
            advice += "Good conditions for normal farming activities."
            actions = ["Continue regular irrigation", "Good time for field operations"]
            
        return {
            'primary_advice': advice,
            'action_items': actions,
            'confidence_score': 0.9
        }
        
    def _generate_market_response(self, entities: Dict, context: Dict,
                                real_time_data: Dict) -> Dict:
        """Generate market-based selling advice"""
        market = real_time_data.get('market', {})
        price = market.get('price', 0)
        change = market.get('change', 'N/A')
        
        crop = entities.get('crop', 'your crop')
        
        advice = f"Current {crop} price: ₹{price}/quintal ({change}). "
        
        if '+' in str(change):
            advice += "Prices are rising. Good time to sell."
            actions = ["Sell immediately if ready", "Check nearby mandis for best rates"]
        else:
            advice += "Prices declining. Consider waiting if possible."
            actions = ["Store safely if possible", "Monitor price trends for 1 week"]
            
        return {
            'primary_advice': advice,
            'action_items': actions,
            'confidence_score': 0.8
        }
        
    def _generate_fertilizer_response(self, entities: Dict, context: Dict,
                                    real_time_data: Dict) -> Dict:
        """Generate fertilizer recommendations"""
        crop = entities.get('crop', 'wheat')
        
        if crop in self.knowledge_base['crops']:
            fertilizer = self.knowledge_base['crops'][crop]['fertilizer']
            advice = f"For {crop}: Apply {fertilizer}. "
            
            weather = real_time_data.get('weather', {}).get('current', {})
            temp = weather.get('temperature', 25)
            
            if temp > 30:
                advice += "High temperature - apply in evening."
                actions = ["Apply after 5 PM", "Water lightly after application"]
            else:
                advice += "Good conditions for fertilizer application."
                actions = ["Apply in morning hours", "Incorporate into soil"]
        else:
            advice = "General fertilizer recommendation: NPK 120:60:60 kg/hectare"
            actions = ["Soil test recommended", "Split application advised"]
            
        return {
            'primary_advice': advice,
            'action_items': actions,
            'confidence_score': 0.85
        }
        
    def _generate_pest_response(self, entities: Dict, context: Dict,
                               real_time_data: Dict) -> Dict:
        """Generate pest management advice"""
        crop = entities.get('crop', 'crop')
        
        advice = f"For {crop} pest management: Regular monitoring essential. "
        
        weather = real_time_data.get('weather', {}).get('current', {})
        humidity = weather.get('humidity', 60)
        
        if humidity > 75:
            advice += "High humidity increases fungal disease risk."
            actions = ["Spray preventive fungicide", "Improve air circulation"]
        else:
            advice += "Current conditions moderate for pest activity."
            actions = ["Weekly field monitoring", "Use pheromone traps"]
            
        return {
            'primary_advice': advice,
            'action_items': actions,
            'warnings': ["Always read pesticide labels", "Use protective
