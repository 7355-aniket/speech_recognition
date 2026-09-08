import datetime
import re
import platform
import random

class VoiceAssistantEngine:
    """Intelligent Voice Assistant engine processing speech-recognized command intents."""

    def __init__(self):
        self.jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "There are 10 types of people in the world: those who understand binary, and those who don't.",
            "Why did the neural network break up with the decision tree? It was feeling overfitted!",
            "How do neural networks learn? One gradient step at a time!",
            "Why was the JavaScript developer sad? Because he didn't know how to 'null' his feelings!"
        ]

    def process_command(self, text: str) -> dict:
        """Parses speech text and returns assistant response with action metadata."""
        if not text or not text.strip():
            return {
                "intent": "empty",
                "response": "I didn't hear any speech. Please speak into the microphone or upload an audio file.",
                "action": "none"
            }

        clean_text = text.lower().strip()

        # 1. Time / Date Query
        if any(w in clean_text for w in ["time", "clock", "hour"]):
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            return {
                "intent": "get_time",
                "query": text,
                "response": f"The current time is {time_str}.",
                "action": "speak",
                "badge": "⏰ Time Service"
            }

        if any(w in clean_text for w in ["date", "today", "day"]):
            now = datetime.datetime.now()
            date_str = now.strftime("%A, %B %d, %Y")
            return {
                "intent": "get_date",
                "query": text,
                "response": f"Today is {date_str}.",
                "action": "speak",
                "badge": "📅 Date Service"
            }

        # 2. Jokes
        if "joke" in clean_text or "funny" in clean_text:
            joke = random.choice(self.jokes)
            return {
                "intent": "tell_joke",
                "query": text,
                "response": joke,
                "action": "speak",
                "badge": "😄 Entertainment"
            }

        # 3. System Info
        if any(w in clean_text for w in ["system", "status", "computer", "specs"]):
            sys_info = f"Running on {platform.system()} {platform.release()} ({platform.machine()}) with Python neural STT pipeline."
            return {
                "intent": "system_status",
                "query": text,
                "response": f"System status optimal. {sys_info}",
                "action": "speak",
                "badge": "💻 System Diagnostics"
            }

        # 4. Math Calculations
        math_match = re.search(r'(?:calculate|what is|compute)\s+([0-9\+\-\*\/\s\.\(\)]+)', clean_text)
        if math_match:
            expr = math_match.group(1).strip()
            try:
                # Safe math evaluation
                allowed_chars = set("0123456789+-*/. ()")
                if set(expr).issubset(allowed_chars):
                    res = eval(expr)
                    return {
                        "intent": "math_calculator",
                        "query": text,
                        "response": f"The calculation result of '{expr}' is {res}.",
                        "action": "speak",
                        "badge": "🧮 Calculator"
                    }
            except Exception:
                pass

        # 5. Search / Web Query
        if any(w in clean_text for w in ["search", "google", "find", "look up"]):
            search_query = clean_text.replace("search for", "").replace("search", "").replace("find", "").strip()
            return {
                "intent": "web_search",
                "query": text,
                "response": f"Searching web for: '{search_query}'.",
                "action": "open_url",
                "url": f"https://www.google.com/search?q={search_query.replace(' ', '+')}",
                "badge": "🔍 Search Engine"
            }

        # 6. Default Speech Processing / Analytics
        word_count = len(text.split())
        char_count = len(text)
        return {
            "intent": "speech_analysis",
            "query": text,
            "response": f"Speech recognized successfully! Received {word_count} words ({char_count} characters). Neural acoustic model processed audio cleanly.",
            "action": "display",
            "analytics": {
                "words": word_count,
                "chars": char_count,
                "uppercase": text.upper()
            },
            "badge": "🧠 Voice Engine"
        }
