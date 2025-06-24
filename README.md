First Iteration of Electra (React Native) 
# Electra ⚡️

A voice-powered trading app built with React Native and Expo, featuring a stunning neural network orb interface.

![Electra Banner](https://img.shields.io/badge/Electra-Voice%20Trading-00D4FF?style=for-the-badge&logo=react&logoColor=white)

## 🎯 Features

- **Voice-First Interface**: Natural language trading commands
- **Neural Network Orb**: Beautiful animated interface with synaptic connections
- **Real-Time Trading**: Execute trades with voice commands
- **Market Analytics**: Live market data and trending stocks
- **Portfolio Management**: Track positions and P&L in real-time
- **Pure Black UI**: OLED-optimized design with electric blue accents

## 🚀 Getting Started

### Prerequisites

- Node.js (v14 or higher)
- Expo CLI
- iOS Simulator (Mac) or Android Emulator

### Installation

1. Clone the repository
```bash
git clone https://github.com/dmeade1/Electra.git
cd Electra
```

2. Install dependencies
```bash
npm install
```

3. Start the development server
```bash
npx expo start
```

4. Run on your device
- Press `i` for iOS simulator
- Press `a` for Android emulator
- Scan QR code with Expo Go app on your phone

## 🎙️ Voice Commands

Try these voice commands:
- "Buy 100 shares of Apple"
- "What's the market doing?"
- "Show my portfolio"
- "Sell 50 shares of Tesla"
- "What's trending today?"
- "Analysis on NVDA"

## 🏗️ Tech Stack

- **React Native** - Cross-platform mobile framework
- **Expo** - Development platform
- **TypeScript** - Type safety
- **React Navigation** - Navigation library
- **Linear Gradient** - Beautiful gradients
- **React Native SVG** - Neural network animations

## 📱 Screens

1. **Voice Screen** - Main interaction with neural orb
2. **Portfolio** - Trading dashboard with positions
3. **Markets** - Real-time market data
4. **Trade** - Execute buy/sell orders
5. **Profile** - User settings and preferences

## 🎨 Design Philosophy

Electra embraces a voice-first approach with visual UI as a fallback. The neural orb represents the AI brain processing your commands, with synaptic connections that pulse with activity.

## 🛠️ Project Structure

```
Electra/
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── ElectraOrb.tsx
│   │   │   ├── GlassCard.tsx
│   │   │   └── NeonButton.tsx
│   │   └── voice/
│   │       └── VoiceWaveform.tsx
│   ├── screens/
│   │   ├── VoiceScreen.tsx
│   │   ├── HomeScreen.tsx
│   │   ├── MarketsScreen.tsx
│   │   ├── TradeScreen.tsx
│   │   └── ProfileScreen.tsx
│   ├── navigation/
│   │   └── TabNavigator.tsx
│   ├── hooks/
│   │   └── useVoiceCommands.ts
│   └── styles/
│       └── theme.ts
└── App.tsx
```

## 🔮 Future Enhancements

- [ ] Real voice recognition integration
- [ ] Live market data API
- [ ] Biometric authentication
- [ ] Push notifications for price alerts
- [ ] Advanced charting with TradingView
- [ ] Social trading features
- [ ] AI-powered trade suggestions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by sci-fi interfaces and neural networks
- Built with React Native and Expo
- Electric blue theme for that cyberpunk trading vibe

---

**Built with ⚡️ by [Drew Meade](https://github.com/dmeade1)**