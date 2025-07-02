# 🤖 Enhanced Dual-Agent Chatbot

A sophisticated conversational AI system that intelligently routes messages to specialized agents based on content classification. The chatbot features emotional support, logical analysis, mixed responses, and crisis detection capabilities.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Key Features

### 🎭 Intelligent Multi-Agent System
- **💙 Therapist Agent**: Provides empathetic emotional support and validation
- **🧠 Logical Agent**: Delivers fact-based analysis and practical solutions
- **🎭 Mixed Agent**: Combines emotional understanding with logical information
- **🆘 Crisis Handler**: Detects and responds to mental health emergencies

### 🔍 Advanced Classification
- Confidence scoring for message classification
- Emotion detection and tracking
- Context-aware routing decisions
- Manual override options for agent selection

### 💾 Conversation Management
- Auto-save conversations with configurable intervals
- Load and resume previous sessions
- Export conversations in multiple formats (TXT, MD, JSON)
- Session analytics and statistics

### 🎨 Enhanced User Experience
- Colored terminal output for better readability
- Customizable response lengths (short/medium/long)
- Temperature control for response creativity
- Real-time confidence display

## 📋 Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Anthropic API key

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/enhanced-dual-agent-chatbot.git
cd enhanced-dual-agent-chatbot
```

### 2. Install uv (if not already installed)

```bash
pip3 install uv
```

### 3. Run the Chatbot

Since the project already includes `pyproject.toml` with all dependencies configured, simply run:

```bash
uv run chatbot.py
```

That's it! uv will automatically handle the virtual environment and install all required dependencies.

### 4. Set Up Environment Variables (First Time Only)

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Anthropic API key:
```env
ANTHROPIC_API_KEY=your_actual_api_key_here
```

## 📦 Dependencies

The project uses `pyproject.toml` for dependency management. If you need to set up a new project from scratch:

```bash
uv init .
uv add python-dotenv langgraph langchain[anthropic] ipykernel
```

Current dependencies:
- `python-dotenv` - Environment variable management
- `langgraph` - Graph-based conversation flow
- `langchain[anthropic]` - LangChain with Anthropic integration
- `ipykernel` - Jupyter kernel support (optional)

## 🎮 Usage

### Basic Usage

```bash
uv run chatbot.py
```

### Example Interactions

1. **Emotional Support**:
   ```
   You: I'm feeling really anxious about my job interview tomorrow
   🔍 Classified as: emotional (confidence: 0.92)
   💭 Emotions detected: anxious, worried
   🧭 Routing to: 💙 Therapist Agent
   
   💙 Therapist: I can really understand why you're feeling anxious...
   ```

2. **Logical Query**:
   ```
   You: What are the main differences between Python and JavaScript?
   🔍 Classified as: logical (confidence: 0.88)
   🧭 Routing to: 🧠 Logical Agent
   
   🧠 Logical: Python and JavaScript are both popular programming languages...
   ```

3. **Mixed Query**:
   ```
   You: I'm stressed about learning programming. Where should I start?
   🔍 Classified as: mixed (confidence: 0.85)
   💭 Emotions detected: stressed, uncertain
   🧭 Routing to: 🎭 Mixed Agent
   
   🎭 Mixed: I understand that learning programming can feel overwhelming...
   ```

### Command Reference

| Command | Description | Example |
|---------|-------------|---------|
| `help` | Display all available commands | `help` |
| `exit` / `quit` | Exit the chatbot | `exit` |
| `clear` | Clear conversation history | `clear` |
| `stats` | Show conversation statistics | `stats` |
| `save [filename]` | Save current conversation | `save interview_chat` |
| `load <filename>` | Load a previous conversation | `load session_20241122_143052.json` |
| `export [format]` | Export conversation (txt/md/json) | `export md` |
| `sessions` | List all saved sessions | `sessions` |
| `config` | Show current configuration | `config` |
| `set temp <value>` | Set response temperature (0.0-1.0) | `set temp 0.8` |
| `set length <value>` | Set response length | `set length long` |
| `toggle colors` | Toggle colored output | `toggle colors` |
| `toggle confidence` | Toggle confidence display | `toggle confidence` |

### Force Agent Override

You can manually select which agent responds:

```bash
force:logical What is machine learning?
force:emotional I need someone to talk to
force:mixed I'm worried about my coding skills
```

## ⚙️ Configuration

### Customizable Settings

You can modify the configuration in `main()`:

```python
config = ChatbotConfig(
    therapist_temperature=0.7,      # Creativity for emotional responses (0.0-1.0)
    logical_temperature=0.3,        # Creativity for logical responses (0.0-1.0)
    max_tokens=1000,               # Maximum response length
    save_conversations=True,        # Auto-save conversations
    show_confidence=True,          # Display classification confidence
    colored_output=True,           # Enable colored terminal output
    auto_save_interval=5,          # Save every N messages
    response_length="medium",      # Default response length (short/medium/long)
    crisis_keywords=[              # Keywords that trigger crisis mode
        "suicide", "kill myself", "end my life", 
        "hurt myself", "self harm", "want to die"
    ]
)
```

### Temperature Settings

- **Lower temperature (0.0-0.3)**: More focused, deterministic responses
- **Medium temperature (0.4-0.7)**: Balanced creativity and coherence
- **Higher temperature (0.8-1.0)**: More creative, varied responses

## 📊 Analytics & Statistics

The chatbot tracks various metrics:

- Total messages exchanged
- Response type distribution (emotional/logical/mixed)
- Crisis detections
- Average classification confidence
- Common detected emotions
- Session duration

View statistics anytime with the `stats` command.

## 💾 Data Storage

### Conversation Files

Conversations are stored in the `conversations/` directory:

```
conversations/
├── session_20241122_143052.json
├── session_20241122_151230.json
├── export_20241122_152145.txt
└── export_20241122_152145.md
```

### File Formats

**JSON Format** (for saving/loading):
```json
{
  "session_id": "session_20241122_143052",
  "start_time": "2024-11-22T14:30:52",
  "end_time": "2024-11-22T15:12:30",
  "messages": [...],
  "stats": {...},
  "config": {...}
}
```

**Export Formats**:
- **TXT**: Plain text conversation
- **MD**: Markdown formatted conversation
- **JSON**: Complete session data

## 🔧 Development

### Project Structure

```
enhanced-dual-agent-chatbot/
├── chatbot.py              # Main application file
├── .env                    # Environment variables (not in repo)
├── pyproject.toml         # Project configuration and dependencies
├── README.md              # This file
```

### Adding New Agents

To add a new specialized agent:

1. Create the agent method:
```python
def _custom_agent(self, state: State):
    # Your agent logic here
    pass
```

2. Add to the graph builder:
```python
graph_builder.add_node("custom", self._custom_agent)
```

3. Update routing logic in `_router()` method

4. Add to classification options

### Extending Crisis Keywords

Add keywords to the configuration:

```python
crisis_keywords = [
    "your", "custom", "keywords", "here"
]
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   ❌ ANTHROPIC_API_KEY not found in environment variables.
   ```
   **Solution**: Ensure `.env` file exists with valid API key

2. **Connection Error**
   ```
   ❌ Error initializing chat model: [error details]
   ```
   **Solution**: Check internet connection and API key validity

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain'
   ```
   **Solution**: Ensure `pyproject.toml` exists and reinstall dependencies:
   ```bash
   uv add python-dotenv langgraph langchain[anthropic]
   ```

4. **Permission Error on Save**
   ```
   ❌ Failed to save conversation: [Permission denied]
   ```
   **Solution**: Ensure write permissions for the `conversations/` directory

### Debug Mode

For verbose output, modify the initialization:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Anthropic's Claude](https://www.anthropic.com/)
- Terminal colors inspired by [colorama](https://pypi.org/project/colorama/)

## 📞 Support & Contact

- **Issues**: Please use the GitHub issue tracker
- **Discussions**: Use GitHub Discussions for questions
- **Crisis Resources**: 
  - US Crisis Hotline: 988
  - Crisis Text Line: Text HOME to 741741
  - International: [befrienders.org](https://www.befrienders.org/)

## 🚧 Roadmap

- [ ] Web interface using Streamlit/Gradio
- [ ] Voice input/output support
- [ ] Multi-language support
- [ ] Integration with external knowledge bases
- [ ] Advanced emotion tracking and mood charts
- [ ] Plugin system for custom agents
- [ ] Docker containerization
- [ ] API endpoint for integration

---

**Remember**: This chatbot is designed to provide support and assistance, but it's not a replacement for professional mental health services. If you're in crisis, please reach out to professional help immediately.