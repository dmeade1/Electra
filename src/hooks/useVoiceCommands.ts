// src/hooks/useVoiceCommands.ts
import { useState, useCallback } from 'react';

interface VoiceCommand {
  command: string;
  action: () => void;
  response: string;
}

interface ParsedCommand {
  action: 'buy' | 'sell' | 'check' | 'show' | 'analyze';
  symbol?: string;
  quantity?: number;
  orderType?: 'market' | 'limit' | 'stop';
  price?: number;
}

export const useVoiceCommands = (navigation: any) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  const [response, setResponse] = useState('');

  // Parse natural language commands
  const parseCommand = (text: string): ParsedCommand | null => {
    const lowerText = text.toLowerCase();
    
    // Buy commands
    const buyMatch = lowerText.match(/buy (\d+) (?:shares? of )?(\w+)/);
    if (buyMatch) {
      return {
        action: 'buy',
        quantity: parseInt(buyMatch[1]),
        symbol: buyMatch[2].toUpperCase(),
        orderType: 'market',
      };
    }

    // Sell commands
    const sellMatch = lowerText.match(/sell (\d+) (?:shares? of )?(\w+)/);
    if (sellMatch) {
      return {
        action: 'sell',
        quantity: parseInt(sellMatch[1]),
        symbol: sellMatch[2].toUpperCase(),
        orderType: 'market',
      };
    }

    // Check price
    if (lowerText.includes('price') || lowerText.includes('quote')) {
      const symbolMatch = lowerText.match(/(?:price|quote) (?:of |for )?(\w+)/);
      if (symbolMatch) {
        return {
          action: 'check',
          symbol: symbolMatch[1].toUpperCase(),
        };
      }
    }

    // Show portfolio
    if (lowerText.includes('portfolio') || lowerText.includes('positions')) {
      return { action: 'show' };
    }

    // Market analysis
    if (lowerText.includes('market') || lowerText.includes('trending')) {
      return { action: 'analyze' };
    }

    return null;
  };

  // Execute voice command
  const executeCommand = useCallback(async (transcription: string) => {
    setIsProcessing(true);
    setLastCommand(transcription);

    const parsed = parseCommand(transcription);
    
    if (!parsed) {
      setResponse("I didn't understand that. Try saying 'Buy 100 shares of Apple' or 'Show my portfolio'");
      setIsProcessing(false);
      return;
    }

    switch (parsed.action) {
      case 'buy':
      case 'sell':
        // Navigate to trade screen with pre-filled data
        navigation.navigate('MainTabs', {
          screen: 'Trade',
          params: {
            action: parsed.action,
            symbol: parsed.symbol,
            quantity: parsed.quantity,
          },
        });
        setResponse(`Opening ${parsed.action} order for ${parsed.quantity} shares of ${parsed.symbol}`);
        break;

      case 'check':
        // Could show a price modal or navigate to symbol details
        setResponse(`Checking price for ${parsed.symbol}...`);
        // Implement price lookup
        break;

      case 'show':
        navigation.navigate('MainTabs', { screen: 'Portfolio' });
        setResponse('Showing your portfolio');
        break;

      case 'analyze':
        navigation.navigate('MainTabs', { screen: 'Markets' });
        setResponse('Opening market analysis');
        break;
    }

    setIsProcessing(false);
  }, [navigation]);

  // Simulated voice responses
  const speakResponse = (text: string) => {
    // In a real app, use text-to-speech here
    console.log('Speaking:', text);
  };

  return {
    isProcessing,
    lastCommand,
    response,
    executeCommand,
    parseCommand,
    speakResponse,
  };
};