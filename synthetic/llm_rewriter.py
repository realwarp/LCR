"""LLM rewriter for natural conversation improvement."""

import random
from typing import List, Dict, Optional
import json
import requests


class LLMRewriter:
    """Optional LLM-based conversation rewriter with fallback."""
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'enabled': False,
            'model': 'llama3',
            'host': 'http://localhost:11434',
            'temperature': 0.7,
            'max_retries': 3,
            'confidence_threshold': 0.8
        }
        self._successes = 0
        self._fallbacks = 0
        self._failures = 0
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        if not self.config.get('enabled', False):
            return False
        
        host = self.config.get('host', 'http://localhost:11434')
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def rewrite_message(
        self,
        original: str,
        context: Dict = None,
        preserve_claim: bool = True,
        seed: int = None
    ) -> Dict:
        """Rewrite a message for naturalness while preserving truth/lie."""
        rng = random.Random(seed)
        
        # If LLM not available or disabled, use template-based variations
        if not self.is_available():
            return {
                'rewritten': self._template_rewrite(original, rng),
                'confidence': 0.5,
                'source': 'template',
                'preserved_claim': True
            }
        
        model = self.config.get('model', 'llama3')
        host = self.config.get('host', 'http://localhost:11434')
        
        prompt = self._build_prompt(original, context, preserve_claim)
        
        for attempt in range(self.config.get('max_retries', 3)):
            try:
                response = requests.post(
                    f"{host}/api/generate",
                    json={
                        'model': model,
                        'prompt': prompt,
                        'temperature': self.config.get('temperature', 0.7),
                        'stream': False
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    rewritten = result.get('response', '').strip()
                    
                    # Validate output
                    validation = self._validate_rewrite(
                        original=original,
                        rewritten=rewritten,
                        context=context,
                        preserve_claim=preserve_claim
                    )
                    
                    if validation['valid']:
                        self._successes += 1
                        return {
                            'rewritten': rewritten,
                            'confidence': validation['confidence'],
                            'source': 'llm',
                            'preserved_claim': validation['preserved_claim']
                        }
                    else:
                        self._fallbacks += 1
                        return {
                            'rewritten': self._template_rewrite(original, rng),
                            'confidence': 0.5,
                            'source': 'template',
                            'preserved_claim': True
                        }
                
            except Exception:
                continue
        
        # All retries failed
        self._failures += 1
        return {
            'rewritten': self._template_rewrite(original, rng),
            'confidence': 0.5,
            'source': 'template',
            'preserved_claim': True
        }
    
    def rewrite_conversation(
        self,
        conversation: List[Dict],
        preserve_liar: bool = True,
        seed: int = None
    ) -> List[Dict]:
        """Rewrite an entire conversation."""
        rng = random.Random(seed)
        
        rewritten = []
        for msg in conversation:
            rewrite_result = self.rewrite_message(
                original=msg['message'],
                context={
                    'speaker': msg['speaker'],
                    'is_liar': msg.get('is_liar', False),
                    'message_type': msg.get('message_type', 'unknown')
                },
                preserve_claim=preserve_liar,
                seed=rng.randint(0, 10000)
            )
            
            rewritten.append({
                'speaker': msg['speaker'],
                'message': rewrite_result['rewritten'],
                'message_type': msg.get('message_type', 'unknown'),
                'fact': msg.get('fact'),
                'timestamp': msg.get('timestamp'),
                'rewrite_source': rewrite_result['source'],
                'rewrite_confidence': rewrite_result['confidence']
            })
        
        return rewritten
    
    def _build_prompt(
        self,
        original: str,
        context: Dict = None,
        preserve_claim: bool = True
    ) -> str:
        """Build prompt for LLM."""
        system = (
            "You are a conversation writer. Rewrite the following message to sound more natural "
            "and conversational while preserving all factual claims and the speaker's intent. "
            "Do not change the truthfulness or introduce new information."
        )
        
        if preserve_claim:
            system += "\nCRITICAL: Preserve the original claim's truth value exactly."
        
        user = f"Original: \"{original}\"\n\nRewritten version:"
        
        return f"{system}\n\n{user}"
    
    def _validate_rewrite(
        self,
        original: str,
        rewritten: str,
        context: Dict,
        preserve_claim: bool
    ) -> Dict:
        """Validate that rewrite preserves necessary elements."""
        confidence = 0.8
        preserved = True
        
        # Basic checks
        if len(rewritten) < len(original) * 0.5:
            confidence = 0.3
            preserved = False
        
        if len(rewritten) > len(original) * 3:
            confidence = 0.6
            preserved = False
        
        return {
            'valid': confidence >= self.config.get('confidence_threshold', 0.8),
            'confidence': confidence,
            'preserved_claim': preserved
        }
    
    def _template_rewrite(self, original: str, rng) -> str:
        """Template-based rewrite as fallback."""
        templates = [
            "{text}",
            "I think {text.lower()}.",
            "Actually, {text.lower()}.",
            "You know what? {text.lower()}.",
            "Let me say it differently: {text.lower()}."
        ]
        
        return rng.choice(templates).format(text=original)
    
    def get_stats(self) -> Dict:
        """Get rewrite statistics."""
        return {
            'successes': self._successes,
            'fallbacks': self._fallbacks,
            'failures': self._failures
        }


def rewrite_conversation_sample(
    conversation: List[Dict],
    enabled: bool = False
) -> List[Dict]:
    """Convenience function to rewrite a conversation."""
    rewriter = LLMRewriter({'enabled': enabled})
    return rewriter.rewrite_conversation(conversation)
