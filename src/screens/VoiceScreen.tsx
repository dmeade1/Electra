// src/screens/VoiceScreen.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  Animated,
  Dimensions,
  StatusBar,
  ScrollView,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { ElectraOrb } from '../components/common/ElectraOrb';
import { theme } from '../styles/theme';

const { height: screenHeight } = Dimensions.get('window');

interface VoiceScreenProps {
  navigation: any;
}

export const VoiceScreen: React.FC<VoiceScreenProps> = ({ navigation }) => {
  const [isListening, setIsListening] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(50)).current;
  const swipeHintAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    // Welcome animation
    Animated.sequence([
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: true,
        }),
        Animated.timing(slideAnim, {
          toValue: 0,
          duration: 1500,
          useNativeDriver: true,
        }),
      ]),
      Animated.delay(1000),
      Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 1000,
          useNativeDriver: true,
        }),
    ]).start(() => {
      setShowWelcome(false);
      // Show main interface
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true,
      }).start();
    });

    // Swipe hint animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(swipeHintAnim, {
          toValue: 1,
          duration: 2000,
          useNativeDriver: true,
        }),
        Animated.timing(swipeHintAnim, {
          toValue: 0,
          duration: 2000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const handleVoicePress = () => {
    setIsListening(!isListening);
    // Voice recognition logic here
  };

  const navigateToDashboard = () => {
    navigation.navigate('MainTabs');
  };

  const voiceCommands = [
    "What's the market doing?",
    "Buy 100 shares of AAPL",
    "Show my portfolio performance",
    "What's trending today?",
    "Execute my morning trades",
    "Analysis on TSLA",
  ];

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      <LinearGradient
        colors={['#000000', '#000000'] as const}
        style={styles.gradient}
      >
        {/* Welcome Message */}
        {showWelcome && (
          <Animated.View
            style={[
              styles.welcomeContainer,
              {
                opacity: fadeAnim,
                transform: [{ translateY: slideAnim }],
              },
            ]}
          >
            <Text style={styles.welcomeText}>Welcome to Electra</Text>
          </Animated.View>
        )}

        {/* Main Interface */}
        {!showWelcome && (
          <Animated.View style={[styles.mainContent, { opacity: fadeAnim }]}>
            {/* Trading Status Bar */}
            <View style={styles.statusBar}>
              <View style={styles.marketStatus}>
                <View style={[styles.statusDot, { backgroundColor: theme.colors.trading.profit }]} />
                <Text style={styles.statusText}>Markets Open</Text>
              </View>
              <Text style={styles.timeText}>9:34 AM EST</Text>
            </View>

            {/* Orb Section */}
            <View style={styles.orbSection}>
              <TouchableOpacity onPress={handleVoicePress} activeOpacity={0.9}>
                <ElectraOrb isListening={isListening} />
              </TouchableOpacity>
              
              <Animated.View style={styles.promptContainer}>
                {isListening ? (
                  <Text style={styles.listeningText}>Listening...</Text>
                ) : (
                  <>
                    <Text style={styles.mainPrompt}>Say Anything</Text>
                    <Text style={styles.subPrompt}>Voice-powered trading at your command</Text>
                  </>
                )}
              </Animated.View>
            </View>

            {/* Voice Command Suggestions */}
            {!isListening && (
              <View style={styles.suggestionsContainer}>
                <Text style={styles.suggestionsTitle}>Try saying:</Text>
                <ScrollView 
                  horizontal 
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={styles.suggestionsList}
                >
                  {voiceCommands.map((command, index) => (
                    <View key={index} style={styles.suggestionChip}>
                      <Text style={styles.suggestionText}>"{command}"</Text>
                    </View>
                  ))}
                </ScrollView>
              </View>
            )}

            {/* Quick Stats */}
            {!isListening && (
              <View style={styles.quickStats}>
                <View style={styles.statItem}>
                  <Text style={styles.statLabel}>Portfolio</Text>
                  <Text style={[styles.statValue, { color: theme.colors.trading.profit }]}>
                    +4.2%
                  </Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statLabel}>Day's P&L</Text>
                  <Text style={[styles.statValue, { color: theme.colors.trading.profit }]}>
                    +$2,847
                  </Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statLabel}>Open Positions</Text>
                  <Text style={styles.statValue}>12</Text>
                </View>
              </View>
            )}

            {/* Swipe Hint */}
            <TouchableOpacity 
              style={styles.swipeHint} 
              onPress={navigateToDashboard}
            >
              <Animated.View
                style={{
                  opacity: swipeHintAnim,
                }}
              >
                <Text style={styles.swipeText}>Swipe up for interactive dashboard</Text>
                <View style={styles.swipeIndicator}>
                  <View style={styles.swipeLine} />
                </View>
              </Animated.View>
            </TouchableOpacity>
          </Animated.View>
        )}
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  gradient: {
    flex: 1,
  },
  welcomeContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  welcomeText: {
    fontSize: 42,
    fontWeight: '300',
    color: '#FFFFFF',
    letterSpacing: 2,
  },
  mainContent: {
    flex: 1,
  },
  statusBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 50,
    paddingBottom: 20,
  },
  marketStatus: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  timeText: {
    color: '#6B7280',
    fontSize: 14,
  },
  orbSection: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingBottom: 60,
  },
  promptContainer: {
    marginTop: 40,
    alignItems: 'center',
  },
  mainPrompt: {
    fontSize: 36,
    fontWeight: '300',
    color: '#FFFFFF',
    marginBottom: 8,
    letterSpacing: 1,
  },
  subPrompt: {
    fontSize: 16,
    color: '#6B7280',
    letterSpacing: 0.5,
  },
  listeningText: {
    fontSize: 28,
    color: theme.colors.primary.main,
    letterSpacing: 1,
  },
  suggestionsContainer: {
    paddingBottom: 24,
  },
  suggestionsTitle: {
    fontSize: 14,
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 12,
  },
  suggestionsList: {
    paddingHorizontal: 24,
  },
  suggestionChip: {
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginHorizontal: 6,
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  suggestionText: {
    color: theme.colors.primary.light,
    fontSize: 14,
  },
  quickStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    marginHorizontal: 24,
    marginBottom: 40,
    paddingVertical: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.05)',
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  statDivider: {
    width: 1,
    height: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  swipeHint: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  swipeText: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 12,
  },
  swipeIndicator: {
    width: 40,
    height: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    borderRadius: 2,
  },
  swipeLine: {
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(255, 255, 255, 0.4)',
    borderRadius: 2,
  },
});