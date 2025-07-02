import os
import json
import datetime
from pathlib import Path
from typing import Annotated, Literal, Optional, Dict, List
import re
from dataclasses import dataclass, asdict
from collections import Counter
import hashlib

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# ANSI color codes for enhanced output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Text colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

# Configuration and settings
@dataclass
class ChatbotConfig:
    therapist_temperature: float = 0.7
    logical_temperature: float = 0.3
    max_tokens: int = 1000
    save_conversations: bool = True
    show_confidence: bool = True
    colored_output: bool = True
    auto_save_interval: int = 5  # Save every N messages
    crisis_keywords: List[str] = None
    response_length: Literal["short", "medium", "long"] = "medium"
    
    def __post_init__(self):
        if self.crisis_keywords is None:
            self.crisis_keywords = [
                "suicide", "kill myself", "end my life", "hurt myself", 
                "self harm", "want to die", "no point living"
            ]

# Enhanced classification with confidence
class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical", "mixed", "crisis"] = Field(
        ...,
        description="Classify the message type"
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the classification"
    )
    detected_emotions: List[str] = Field(
        default_factory=list,
        description="List of detected emotions if any"
    )
    requires_followup: bool = Field(
        default=False,
        description="Whether this message likely requires follow-up questions"
    )

# Conversation analytics
@dataclass
class ConversationStats:
    total_messages: int = 0
    emotional_responses: int = 0
    logical_responses: int = 0
    mixed_responses: int = 0
    crisis_detections: int = 0
    avg_confidence: float = 0.0
    session_duration: float = 0.0
    common_emotions: Dict[str, int] = None
    
    def __post_init__(self):
        if self.common_emotions is None:
            self.common_emotions = {}

# Enhanced state
class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None
    next: str | None
    confidence: float | None
    reasoning: str | None
    detected_emotions: List[str] | None
    conversation_context: Dict | None
    user_preferences: Dict | None
    session_stats: Dict | None

class EnhancedChatbot:
    def __init__(self, config: ChatbotConfig = None):
        self.config = config or ChatbotConfig()
        self.conversations_dir = Path("conversations")
        self.conversations_dir.mkdir(exist_ok=True)
        self.current_session_id = None
        self.session_start_time = None
        self.conversation_history = []
        self.stats = ConversationStats()
        
        # Load environment and initialize LLM
        self._initialize_llm()
        self._build_graph()
        
    def _initialize_llm(self):
        """Initialize the language model with error handling"""
        load_dotenv()
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            self._print_error("ANTHROPIC_API_KEY not found in environment variables.")
            self._print_info("Please create a .env file with your Anthropic API key:")
            self._print_info("ANTHROPIC_API_KEY=your_actual_api_key_here")
            exit(1)

        api_key = api_key.strip()
        self._print_success(f"API key loaded: {api_key[:15]}...{api_key[-5:]}")

        try:
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=api_key,
                max_tokens=self.config.max_tokens
            )
            
            # Test connection
            test_response = self.llm.invoke("Test connection - respond with 'OK'")
            self._print_success(f"Connection test successful: {test_response.content}")
            
        except Exception as e:
            self._print_error(f"Error initializing chat model: {e}")
            exit(1)

    def _build_graph(self):
        """Build the conversation flow graph"""
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("classifier", self._classify_message)
        graph_builder.add_node("crisis_handler", self._handle_crisis)
        graph_builder.add_node("router", self._router)
        graph_builder.add_node("therapist", self._therapist_agent)
        graph_builder.add_node("logical", self._logical_agent)
        graph_builder.add_node("mixed", self._mixed_agent)
        graph_builder.add_node("analytics", self._update_analytics)

        # Add edges
        graph_builder.add_edge(START, "classifier")
        
        # Conditional routing from classifier
        graph_builder.add_conditional_edges(
            "classifier",
            lambda state: "crisis_handler" if state.get("message_type") == "crisis" else "router"
        )
        
        graph_builder.add_edge("crisis_handler", "analytics")
        graph_builder.add_edge("router", "analytics")
        
        # Conditional routing from router
        graph_builder.add_conditional_edges(
            "router",
            lambda state: state.get("next"),
            {
                "therapist": "therapist", 
                "logical": "logical", 
                "mixed": "mixed"
            }
        )

        graph_builder.add_edge("therapist", "analytics")
        graph_builder.add_edge("logical", "analytics")
        graph_builder.add_edge("mixed", "analytics")
        graph_builder.add_edge("analytics", END)

        self.graph = graph_builder.compile()

    def _get_message_content(self, message):
        """Safely extract content from a message object"""
        if isinstance(message, dict):
            return message.get("content", "")
        elif hasattr(message, "content"):
            return message.content
        else:
            return str(message)

    def _classify_message(self, state: State):
        """Enhanced message classification with confidence and emotion detection"""
        try:
            # Get the last message content safely
            last_message = state["messages"][-1]
            message_content = self._get_message_content(last_message)
            
            classifier_llm = self.llm.with_structured_output(MessageClassifier)

            # Check for crisis keywords first
            message_lower = message_content.lower()
            crisis_detected = any(keyword in message_lower for keyword in self.config.crisis_keywords)
            
            if crisis_detected:
                return {
                    "message_type": "crisis",
                    "confidence": 1.0,
                    "reasoning": "Crisis keywords detected",
                    "detected_emotions": ["distress", "crisis"],
                    "next": "crisis_handler"
                }

            result = classifier_llm.invoke([
                SystemMessage(content="""Classify the user message with confidence and reasoning:
                    - 'emotional': emotional support, therapy, feelings, personal problems
                    - 'logical': facts, information, logical analysis, practical solutions  
                    - 'mixed': requires both emotional support AND logical information
                    - 'crisis': immediate mental health concern (but this should be rare as crisis keywords are pre-screened)
                    
                    Also detect emotions like: happy, sad, angry, anxious, confused, excited, frustrated, etc.
                    Provide confidence (0-1) and brief reasoning for your classification.
                    """),
                HumanMessage(content=message_content)
            ])
            
            self._print_classification(result)
            
            return {
                "message_type": result.message_type,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "detected_emotions": result.detected_emotions
            }
            
        except Exception as e:
            self._print_error(f"Error in classification: {e}")
            return {
                "message_type": "logical",
                "confidence": 0.5,
                "reasoning": "Fallback due to classification error",
                "detected_emotions": []
            }

    def _handle_crisis(self, state: State):
        """Special handling for crisis situations"""
        try:
            crisis_response = """I'm very concerned about what you're sharing. Your feelings are important, and you deserve support right now.

🆘 **Immediate Help Available:**
• **Crisis Hotline**: 988 (US) - Available 24/7
• **Crisis Text Line**: Text HOME to 741741
• **International**: befrienders.org

Please reach out to these resources or a trusted person in your life. You don't have to go through this alone.

Would you like to talk about what's making you feel this way, or would you prefer information about other support resources?"""

            self.stats.crisis_detections += 1
            
            return {
                "messages": [AIMessage(content=crisis_response)],
                "next": "analytics"
            }
            
        except Exception as e:
            self._print_error(f"Error in crisis handler: {e}")
            return {
                "messages": [AIMessage(content="I'm concerned about you and want to help. Please consider reaching out to a crisis helpline: 988 (US) or your local emergency services.")]
            }

    def _router(self, state: State):
        """Enhanced routing with mixed mode support"""
        message_type = state.get("message_type", "logical")
        
        route_map = {
            "emotional": "therapist",
            "logical": "logical", 
            "mixed": "mixed"
        }
        
        next_node = route_map.get(message_type, "logical")
        self._print_routing(next_node, state.get("confidence", 0))
        
        return {"next": next_node}

    def _get_response_length_instruction(self):
        """Get response length instruction based on config"""
        length_map = {
            "short": "Keep responses concise, 1-2 paragraphs maximum.",
            "medium": "Provide moderate detail, 2-4 paragraphs.",
            "long": "Give comprehensive responses with full detail and examples."
        }
        return length_map.get(self.config.response_length, length_map["medium"])

    def _get_conversation_context(self, state: State):
        """Generate conversation context for agents"""
        recent_messages = state["messages"][-5:]  # Last 5 messages for context
        context = ""
        
        if len(recent_messages) > 1:
            context = "\n\nRecent conversation context:\n"
            for msg in recent_messages[:-1]:  # Exclude current message
                content = self._get_message_content(msg)
                if isinstance(msg, HumanMessage):
                    role = "User"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                else:
                    role = "System"
                context += f"{role}: {content[:100]}...\n"
        
        return context

    def _therapist_agent(self, state: State):
        """Enhanced therapist with context awareness"""
        try:
            last_message = state["messages"][-1]
            message_content = self._get_message_content(last_message)
            context = self._get_conversation_context(state)
            emotions = state.get("detected_emotions", [])
            emotion_context = f"\nDetected emotions: {', '.join(emotions)}" if emotions else ""
            
            system_prompt = f"""You are a compassionate, skilled therapist. {self._get_response_length_instruction()}
            
            Focus on emotional support and validation. Show empathy and help the user process their feelings.
            Ask thoughtful questions to facilitate deeper emotional exploration.
            Use therapeutic techniques like reflection, validation, and gentle challenging when appropriate.
            
            {emotion_context}
            {context}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]
            
            # Use therapist-specific temperature
            therapist_llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY").strip(),
                max_tokens=self.config.max_tokens,
                temperature=self.config.therapist_temperature
            )
            
            reply = therapist_llm.invoke(messages)
            self.stats.emotional_responses += 1
            
            return {"messages": [reply]}
            
        except Exception as e:
            self._print_error(f"Error in therapist agent: {e}")
            return {"messages": [AIMessage(content="I'm here to support you emotionally. Could you help me understand what you're feeling right now?")]}

    def _logical_agent(self, state: State):
        """Enhanced logical agent with context awareness"""
        try:
            last_message = state["messages"][-1]
            message_content = self._get_message_content(last_message)
            context = self._get_conversation_context(state)
            
            system_prompt = f"""You are a logical, analytical assistant. {self._get_response_length_instruction()}
            
            Provide clear, fact-based responses with evidence and reasoning.
            Be direct, precise, and focus on practical solutions.
            Use structured thinking and break down complex problems logically.
            Avoid emotional language and focus on objective analysis.
            
            {context}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]
            
            # Use logical-specific temperature
            logical_llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY").strip(),
                max_tokens=self.config.max_tokens,
                temperature=self.config.logical_temperature
            )
            
            reply = logical_llm.invoke(messages)
            self.stats.logical_responses += 1
            
            return {"messages": [reply]}
            
        except Exception as e:
            self._print_error(f"Error in logical agent: {e}")
            return {"messages": [AIMessage(content="I encountered an error processing your logical query. Please try rephrasing your question.")]}

    def _mixed_agent(self, state: State):
        """New mixed agent for questions requiring both emotional and logical responses"""
        try:
            last_message = state["messages"][-1]
            message_content = self._get_message_content(last_message)
            context = self._get_conversation_context(state)
            emotions = state.get("detected_emotions", [])
            emotion_context = f"\nDetected emotions: {', '.join(emotions)}" if emotions else ""
            
            system_prompt = f"""You are a balanced assistant combining empathy with logic. {self._get_response_length_instruction()}
            
            Address both the emotional AND informational aspects of the user's message:
            1. First, acknowledge and validate any emotions or feelings
            2. Then provide logical, factual information or practical solutions
            3. Connect the emotional and logical aspects naturally
            
            Be warm but also informative. Show you understand their feelings while providing helpful information.
            
            {emotion_context}
            {context}
            """

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message_content)
            ]
            
            reply = self.llm.invoke(messages)
            self.stats.mixed_responses += 1
            
            return {"messages": [reply]}
            
        except Exception as e:
            self._print_error(f"Error in mixed agent: {e}")
            return {"messages": [AIMessage(content="I understand this is both an emotional and practical matter for you. Let me try to help with both aspects.")]}

    def _update_analytics(self, state: State):
        """Update conversation analytics"""
        self.stats.total_messages += 1
        
        # Update confidence average
        if state.get("confidence"):
            current_avg = self.stats.avg_confidence
            total = self.stats.total_messages
            self.stats.avg_confidence = ((current_avg * (total - 1)) + state["confidence"]) / total
        
        # Update emotions
        emotions = state.get("detected_emotions", [])
        for emotion in emotions:
            self.stats.common_emotions[emotion] = self.stats.common_emotions.get(emotion, 0) + 1
        
        # Auto-save if enabled
        if self.config.save_conversations and self.stats.total_messages % self.config.auto_save_interval == 0:
            self._auto_save_session()
        
        return state

    def _print_colored(self, text: str, color: str = Colors.WHITE, bold: bool = False):
        """Print colored text if colors are enabled"""
        if self.config.colored_output:
            style = Colors.BOLD if bold else ""
            print(f"{style}{color}{text}{Colors.RESET}")
        else:
            print(text)

    def _print_success(self, text: str):
        self._print_colored(f"✅ {text}", Colors.GREEN)

    def _print_error(self, text: str):
        self._print_colored(f"❌ {text}", Colors.RED)

    def _print_info(self, text: str):
        self._print_colored(f"ℹ️  {text}", Colors.BLUE)

    def _print_warning(self, text: str):
        self._print_colored(f"⚠️  {text}", Colors.YELLOW)

    def _print_classification(self, result):
        """Print classification results with color coding"""
        if self.config.show_confidence:
            confidence_color = Colors.GREEN if result.confidence > 0.8 else Colors.YELLOW if result.confidence > 0.6 else Colors.RED
            self._print_colored(f"🔍 Classified as: {result.message_type} (confidence: {result.confidence:.2f})", confidence_color)
            if result.detected_emotions:
                self._print_colored(f"💭 Emotions detected: {', '.join(result.detected_emotions)}", Colors.MAGENTA)

    def _print_routing(self, agent: str, confidence: float):
        """Print routing information"""
        agent_emojis = {"therapist": "💙", "logical": "🧠", "mixed": "🎭", "crisis": "🆘"}
        emoji = agent_emojis.get(agent, "🤖")
        self._print_colored(f"🧭 Routing to: {emoji} {agent.title()} Agent", Colors.CYAN)

    def _create_session_id(self):
        """Create unique session ID"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"

    def _save_conversation(self, filename: str = None):
        """Save conversation to file"""
        if not filename:
            filename = f"{self.current_session_id}.json"
        
        filepath = self.conversations_dir / filename
        
        conversation_data = {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "end_time": datetime.datetime.now().isoformat(),
            "messages": self.conversation_history,
            "stats": asdict(self.stats),
            "config": asdict(self.config)
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            self._print_success(f"Conversation saved to {filepath}")
            return str(filepath)
        except Exception as e:
            self._print_error(f"Failed to save conversation: {e}")
            return None

    def _load_conversation(self, filename: str):
        """Load conversation from file"""
        filepath = self.conversations_dir / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_session_id = data["session_id"]
            self.conversation_history = data["messages"]
            
            # Restore stats
            stats_data = data.get("stats", {})
            self.stats = ConversationStats(**stats_data)
            
            self._print_success(f"Conversation loaded from {filepath}")
            self._print_info(f"Loaded {len(self.conversation_history)} messages")
            return True
            
        except Exception as e:
            self._print_error(f"Failed to load conversation: {e}")
            return False

    def _auto_save_session(self):
        """Auto-save current session"""
        if self.current_session_id:
            self._save_conversation()

    def _export_conversation(self, format_type: str = "txt"):
        """Export conversation in different formats"""
        if not self.conversation_history:
            self._print_warning("No conversation to export")
            return None
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "txt":
            filename = f"export_{timestamp}.txt"
            filepath = self.conversations_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Conversation Export - {datetime.datetime.now()}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for msg in self.conversation_history:
                        role = "You" if msg["role"] == "user" else "Assistant"
                        f.write(f"{role}: {msg['content']}\n\n")
                
                self._print_success(f"Conversation exported to {filepath}")
                return str(filepath)
                
            except Exception as e:
                self._print_error(f"Failed to export conversation: {e}")
                return None
        
        elif format_type.lower() == "md":
            filename = f"export_{timestamp}.md"
            filepath = self.conversations_dir / filename
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Conversation Export\n\n")
                    f.write(f"**Date:** {datetime.datetime.now()}\n\n")
                    f.write(f"**Messages:** {len(self.conversation_history)}\n\n")
                    f.write("---\n\n")
                    
                    for msg in self.conversation_history:
                        if msg["role"] == "user":
                            f.write(f"## You\n\n{msg['content']}\n\n")
                        else:
                            f.write(f"## Assistant\n\n{msg['content']}\n\n")
                
                self._print_success(f"Conversation exported to {filepath}")
                return str(filepath)
                
            except Exception as e:
                self._print_error(f"Failed to export conversation: {e}")
                return None

    def _show_stats(self):
        """Display conversation statistics"""
        if self.stats.total_messages == 0:
            self._print_info("No messages in current session")
            return
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 CONVERSATION STATISTICS{Colors.RESET}")
        print("=" * 40)
        print(f"Total Messages: {self.stats.total_messages}")
        print(f"Emotional Responses: {self.stats.emotional_responses}")
        print(f"Logical Responses: {self.stats.logical_responses}")
        print(f"Mixed Responses: {self.stats.mixed_responses}")
        print(f"Crisis Detections: {self.stats.crisis_detections}")
        print(f"Average Confidence: {self.stats.avg_confidence:.2f}")
        
        if self.session_start_time:
            duration = datetime.datetime.now() - self.session_start_time
            print(f"Session Duration: {duration}")
        
        if self.stats.common_emotions:
            print(f"\nTop Emotions Detected:")
            sorted_emotions = sorted(self.stats.common_emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_emotions[:5]:
                print(f"  {emotion}: {count}")
        
        print("=" * 40 + "\n")

    def _show_help(self):
        """Display help information"""
        help_text = f"""
{Colors.BOLD}{Colors.CYAN}📖 CHATBOT COMMANDS{Colors.RESET}
{Colors.BLUE}{'=' * 50}{Colors.RESET}

{Colors.BOLD}Basic Commands:{Colors.RESET}
  help              - Show this help message
  exit/quit         - Exit the chatbot
  clear             - Clear conversation history
  stats             - Show conversation statistics

{Colors.BOLD}Session Management:{Colors.RESET}
  save [filename]   - Save conversation (optional filename)
  load <filename>   - Load previous conversation
  export [format]   - Export conversation (txt/md/json)
  sessions          - List saved sessions

{Colors.BOLD}Configuration:{Colors.RESET}
  config            - Show current configuration
  set temp <val>    - Set response creativity (0.0-1.0)
  set length <val>  - Set response length (short/medium/long)
  toggle colors     - Toggle colored output
  toggle confidence - Toggle confidence display

{Colors.BOLD}Overrides:{Colors.RESET}
  force:logical     - Force logical response
  force:emotional   - Force emotional response
  force:mixed       - Force mixed response

{Colors.BLUE}{'=' * 50}{Colors.RESET}
        """
        print(help_text)

    def _list_sessions(self):
        """List available saved sessions"""
        json_files = list(self.conversations_dir.glob("*.json"))
        
        if not json_files:
            self._print_info("No saved sessions found")
            return
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}💾 SAVED SESSIONS{Colors.RESET}")
        print("=" * 40)
        
        for file in sorted(json_files):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                start_time = data.get("start_time", "Unknown")
                message_count = len(data.get("messages", []))
                
                print(f"📁 {file.name}")
                print(f"   Started: {start_time}")
                print(f"   Messages: {message_count}")
                print()
                
            except Exception as e:
                print(f"❌ Error reading {file.name}: {e}")
        
        print("=" * 40 + "\n")

    def _show_config(self):
        """Display current configuration"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}⚙️  CURRENT CONFIGURATION{Colors.RESET}")
        print("=" * 40)
        print(f"Therapist Temperature: {self.config.therapist_temperature}")
        print(f"Logical Temperature: {self.config.logical_temperature}")
        print(f"Max Tokens: {self.config.max_tokens}")
        print(f"Response Length: {self.config.response_length}")
        print(f"Save Conversations: {self.config.save_conversations}")
        print(f"Show Confidence: {self.config.show_confidence}")
        print(f"Colored Output: {self.config.colored_output}")
        print(f"Auto-save Interval: {self.config.auto_save_interval}")
        print("=" * 40 + "\n")

    def _handle_command(self, user_input: str):
        """Handle special commands"""
        parts = user_input.lower().split()
        command = parts[0]
        
        if command in ["exit", "quit"]:
            return "exit"
        elif command == "help":
            self._show_help()
            return "continue"
        elif command == "clear":
            self.conversation_history = []
            self.stats = ConversationStats()
            self._print_success("Conversation history cleared!")
            return "continue"
        elif command == "stats":
            self._show_stats()
            return "continue"
        elif command == "config":
            self._show_config()
            return "continue"
        elif command == "sessions":
            self._list_sessions()
            return "continue"
        elif command == "save":
            filename = parts[1] if len(parts) > 1 else None
            self._save_conversation(filename)
            return "continue"
        elif command == "load" and len(parts) > 1:
            if self._load_conversation(parts[1]):
                return "loaded"
            return "continue"
        elif command == "export":
            format_type = parts[1] if len(parts) > 1 else "txt"
            self._export_conversation(format_type)
            return "continue"
        elif command == "set" and len(parts) >= 3:
            return self._handle_config_change(parts[1], parts[2])
        elif command == "toggle" and len(parts) > 1:
            return self._handle_toggle(parts[1])
        
        return None

    def _handle_config_change(self, setting: str, value: str):
        """Handle configuration changes"""
        try:
            if setting == "temp":
                temp_val = float(value)
                if 0.0 <= temp_val <= 1.0:
                    self.config.therapist_temperature = temp_val
                    self.config.logical_temperature = max(0.1, temp_val - 0.2)
                    self._print_success(f"Temperature set to {temp_val}")
                else:
                    self._print_error("Temperature must be between 0.0 and 1.0")
            elif setting == "length":
                if value in ["short", "medium", "long"]:
                    self.config.response_length = value
                    self._print_success(f"Response length set to {value}")
                else:
                    self._print_error("Length must be short, medium, or long")
            else:
                self._print_error(f"Unknown setting: {setting}")
        except ValueError:
            self._print_error("Invalid value provided")
        
        return "continue"

    def _handle_toggle(self, setting: str):
        """Handle toggle commands"""
        if setting == "colors":
            self.config.colored_output = not self.config.colored_output
            status = "enabled" if self.config.colored_output else "disabled"
            self._print_success(f"Colored output {status}")
        elif setting == "confidence":
            self.config.show_confidence = not self.config.show_confidence
            status = "enabled" if self.config.show_confidence else "disabled"
            self._print_success(f"Confidence display {status}")
        else:
            self._print_error(f"Unknown toggle setting: {setting}")
        
        return "continue"

    def _check_for_override(self, user_input: str):
        """Check for force override commands"""
        lower_input = user_input.lower()
        if lower_input.startswith("force:"):
            override_type = lower_input.split(":", 1)[1].strip()
            actual_message = user_input.split(":", 1)[1].strip()
            
            if override_type in ["logical", "emotional", "mixed"]:
                return override_type, actual_message
        
        return None, user_input

    def run_chatbot(self):
        """Main chatbot loop with all enhanced features"""
        # Initialize session
        self.current_session_id = self._create_session_id()
        self.session_start_time = datetime.datetime.now()
        
        # Welcome message
        welcome_msg = f"""
{Colors.BOLD}{Colors.CYAN}🤖 ENHANCED DUAL-AGENT CHATBOT{Colors.RESET}
{Colors.BLUE}{'=' * 60}{Colors.RESET}

{Colors.BOLD}🎭 Three Response Modes:{Colors.RESET}
💙 {Colors.BLUE}Therapist Mode{Colors.RESET}: Emotional support and feelings
🧠 {Colors.GREEN}Logical Mode{Colors.RESET}: Facts, information, and analysis  
🎭 {Colors.MAGENTA}Mixed Mode{Colors.RESET}: Both emotional support AND logical info
🆘 {Colors.RED}Crisis Mode{Colors.RESET}: Special handling for mental health concerns

{Colors.BOLD}✨ Enhanced Features:{Colors.RESET}
• Confidence scoring and emotion detection
• Conversation saving and loading  
• Export to multiple formats
• Real-time analytics and statistics
• Customizable response styles
• Crisis detection and resources

Type '{Colors.YELLOW}help{Colors.RESET}' for all commands, '{Colors.YELLOW}exit{Colors.RESET}' to quit
Session ID: {Colors.CYAN}{self.current_session_id}{Colors.RESET}
{Colors.BLUE}{'=' * 60}{Colors.RESET}
        """
        
        print(welcome_msg)
        
        state = {
            "messages": [], 
            "message_type": None, 
            "next": None,
            "conversation_context": {},
            "user_preferences": {},
            "session_stats": {}
        }

        while True:
            try:
                # Get user input with prompt styling
                if self.config.colored_output:
                    user_input = input(f"{Colors.BOLD}{Colors.YELLOW}You: {Colors.RESET}").strip()
                else:
                    user_input = input("You: ").strip()
                
                if not user_input:
                    self._print_warning("Please enter a message.")
                    continue

                # Handle commands
                command_result = self._handle_command(user_input)
                if command_result == "exit":
                    break
                elif command_result in ["continue", "loaded"]:
                    if command_result == "loaded":
                        # Rebuild state from loaded conversation
                        state["messages"] = []
                        for msg in self.conversation_history:
                            if msg["role"] == "user":
                                state["messages"].append(HumanMessage(content=msg["content"]))
                            else:
                                state["messages"].append(AIMessage(content=msg["content"]))
                    continue

                # Check for override commands
                override_type, actual_message = self._check_for_override(user_input)
                
                # Add user message to state and history
                user_message = HumanMessage(content=actual_message)
                state["messages"] = state.get("messages", []) + [user_message]
                self.conversation_history.append({"role": "user", "content": actual_message})

                # Handle manual override
                if override_type:
                    self._print_info(f"Manual override: {override_type} mode")
                    
                    if override_type == "logical":
                        result = self._logical_agent(state)
                    elif override_type == "emotional":
                        result = self._therapist_agent(state)
                    elif override_type == "mixed":
                        result = self._mixed_agent(state)
                    
                    # Update state and display response
                    state.update(result)
                    self._update_analytics(state)
                    
                else:
                    # Process through the graph normally
                    try:
                        result_state = self.graph.invoke(state)
                        state.update(result_state)
                    except Exception as e:
                        self._print_error(f"Error processing message: {e}")
                        continue

                # Display assistant response
                if state.get("messages") and len(state["messages"]) > 0:
                    last_message = state["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        # Add to conversation history
                        self.conversation_history.append({"role": "assistant", "content": last_message.content})
                        
                        # Determine agent type for display
                        agent_type = state.get("message_type", "logical")
                        agent_display = {
                            "emotional": f"{Colors.BLUE}💙 Therapist{Colors.RESET}",
                            "logical": f"{Colors.GREEN}🧠 Logical{Colors.RESET}",
                            "mixed": f"{Colors.MAGENTA}🎭 Mixed{Colors.RESET}",
                            "crisis": f"{Colors.RED}🆘 Crisis Support{Colors.RESET}"
                        }.get(agent_type, f"{Colors.WHITE}🤖 Assistant{Colors.RESET}")
                        
                        # Display response with styling
                        if self.config.colored_output:
                            print(f"\n{agent_display}: {last_message.content}\n")
                        else:
                            agent_simple = {
                                "emotional": "💙 Therapist",
                                "logical": "🧠 Logical", 
                                "mixed": "🎭 Mixed",
                                "crisis": "🆘 Crisis Support"
                            }.get(agent_type, "🤖 Assistant")
                            print(f"\n{agent_simple}: {last_message.content}\n")

            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
                break
            except Exception as e:
                self._print_error(f"An unexpected error occurred: {e}")
                self._print_info("Please try again or type 'exit' to quit.")

        # Save session before exit
        if self.config.save_conversations and self.conversation_history:
            save_choice = input(f"\n{Colors.YELLOW}Save this conversation? (y/n): {Colors.RESET}").strip().lower()
            if save_choice in ['y', 'yes', '']:
                self._save_conversation()

        # Final statistics
        if self.stats.total_messages > 0:
            print(f"\n{Colors.BOLD}{Colors.CYAN}📊 SESSION SUMMARY{Colors.RESET}")
            print("=" * 30)
            print(f"Messages exchanged: {self.stats.total_messages}")
            print(f"Session duration: {datetime.datetime.now() - self.session_start_time if self.session_start_time else 'Unknown'}")
            print(f"Average confidence: {self.stats.avg_confidence:.2f}")

        print(f"\n{Colors.BOLD}{Colors.GREEN}👋 Thank you for using Enhanced Dual-Agent Chatbot!{Colors.RESET}")
        print(f"{Colors.CYAN}Take care and remember - support is always available when you need it.{Colors.RESET}\n")


def main():
    """Main entry point with configuration options"""
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 Initializing Enhanced Dual-Agent Chatbot...{Colors.RESET}")
    
    # You can customize the configuration here
    config = ChatbotConfig(
        therapist_temperature=0.7,      # Higher creativity for emotional responses
        logical_temperature=0.3,        # Lower creativity for factual responses  
        max_tokens=1000,               # Response length limit
        save_conversations=True,        # Auto-save conversations
        show_confidence=True,          # Show classification confidence
        colored_output=True,           # Enable colored terminal output
        auto_save_interval=5,          # Save every 5 messages
        response_length="medium"       # Default response length
    )
    
    try:
        chatbot = EnhancedChatbot(config)
        chatbot.run_chatbot()
    except Exception as e:
        print(f"{Colors.RED}❌ Failed to initialize chatbot: {e}{Colors.RESET}")
        print(f"{Colors.YELLOW}Please check your environment setup and try again.{Colors.RESET}")


if __name__ == "__main__":
    main()