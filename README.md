```markdown
# MahsaAI

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Telegram](https://img.shields.io/badge/Telegram-Bot-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A smart Telegram bot with the personality of Mahsa, a 17-year-old student. MahsaAI answers academic questions, generates and analyzes images, processes voice messages, and chats in a friendly, humorous way. Published by **DIGI-X**.

## Features

- 🧠 Answers academic questions (Math, Physics, Chemistry, etc.) with clear explanations.
- 📸 Generates images from text prompts.
- 🖼 Analyzes images (scientific, text-based, or random).
- 🎙 Processes Persian voice messages for text-based interaction.
- 💬 Maintains conversation history for personalized responses.
- 😄 Adjusts tone dynamically based on user mood (formal, friendly, or playful).
- 🔐 User authentication via phone number and Telegram channel membership.
- 📊 Tracks user statistics (images, messages, voice inputs).
- ⏰ Daily reset for usage limits.

## Prerequisites

- Python 3.8 or higher
- Telegram Bot Token (obtain from [BotFather](https://t.me/BotFather))
- Gemini API Key (for advanced AI features)
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MahsaAI.git
   cd MahsaAI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your bot token and API key:
   ```env
   TELEGRAM_BOT_TOKEN=your-telegram-bot-token
   GEMINI_API_KEY=your-gemini-api-key
   ```

4. Run the bot:
   ```bash
   python mahsa.py
   ```

## Usage

1. Start the bot with `/start`.
2. Ask academic questions (e.g., "Solve the equation x^2 + 2x - 3 = 0").
3. Request image generation (e.g., "Generate a cute cat image").
4. Send an image with a caption starting with `.` in groups for analysis (e.g., `.Analyze this math formula`).
5. Send a Persian voice message for text processing.
6. Use `/info` for stats, `/help` for guidance, `/clear` to reset memory, `/clearall` (admin only), or `/bug` to report issues.

## Authentication

- Users must join the Telegram channel: [@DIGI_X](https://t.me/DIGI_X).
- Phone number verification is required to access the bot.

## Project Structure

```
MahsaAI/
├── mahsa.py              # Main bot script
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Submit a pull request with clear documentation.
4. Ensure code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines.

## License

This project is licensed under the [MIT License](LICENSE). See the `LICENSE` file for details.

## Publisher

**DIGI-X**  
- Telegram Channel: [@DIGI_X](https://t.me/DIGI_X)  
- Telegram Username: [@velovpn](https://t.me/velovpn)

## Acknowledgments

- Powered by [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot).
- Image generation and AI via [g4f](https://github.com/xtekky/gpt4free).
- Voice processing with [speech_recognition](https://github.com/Uberi/speech_recognition).
- Audio handling with [pydub](https://github.com/jiaaro/pydub).
- Mathematical processing with [sympy](https://github.com/sympy/sympy).

## Contact

For bug reports or feature requests, contact DIGI-X via:  
- Telegram: [@velovpn](https://t.me/velovpn)  
- Channel: [@DIGI_X](https://t.me/DIGI_X)
```
