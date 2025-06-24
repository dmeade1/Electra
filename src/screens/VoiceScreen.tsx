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
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import Svg, { Circle, Line, G, Defs, RadialGradient, Stop } from 'react-native-svg';
import { ElectraOrb } from '../components/common/ElectraOrb';
import { theme } from '../styles/theme';

const { height: screenHeight, width: screenWidth } = Dimensions.get('window');

interface VoiceScreenProps {
  navigation: any;
}

// Background neural network component
const NeuralBackground: React.FC = () => {
  const backgroundNodes = useRef<Array<{ x: number; y: number; connections: number[] }>>([]);
  
  // Generate background neural network
  if (backgroundNodes.current.length === 0) {
    for (let i = 0; i < 15; i++) {
      backgroundNodes.current.push({
        x: Math.random() * screenWidth,
        y: Math.random() * screenHeight,
        connections: [],
      });
    }
    
    // Create connections
    backgroundNodes.current.forEach((node, index) => {
      const connectionCount = 2 + Math.floor(Math.random() * 2);
      for (let i = 0; i < connectionCount; i++) {
        const targetIndex = Math.floor(Math.random() * backgroundNodes.current.length);
        if (targetIndex !== index) {
          node.connections.push(targetIndex);
        }
      }
    });
  }

  return (
    <View style={StyleSheet.absoluteFillObject}>
      <Svg width={screenWidth} height={screenHeight} style={StyleSheet.absoluteFillObject}>
        <Defs>
          <RadialGradient id="bgNodeGlow" cx="50%" cy="50%">
            <Stop offset="0%" stopColor="#0030FF" stopOpacity="0.3" />
            <Stop offset="100%" stopColor="#0020FF" stopOpacity="0" />
          </RadialGradient>
        </Defs>
        
        {/* Connections */}
        <G opacity={0.1}>
          {backgroundNodes.current.map((node, index) => 
            node.connections.map((targetIndex, connIndex) => {
              const target = backgroundNodes.current[targetIndex];
              return (
                <Line
                  key={`${index}-${connIndex}`}
                  x1={node.x}
                  y1={node.y}
                  x2={target.x}
                  y2={target.y}
                  stroke="#0030FF"
                  strokeWidth={0.5}
                />
              );
            })
          )}
        </G>
        
        {/* Nodes */}
        <G opacity={0.2}>
          {backgroundNodes.current.map((node, index) => (
            <G key={index}>
              <Circle
                cx={node.x}
                cy={node.y}
                r={15}
                fill="url(#bgNodeGlow)"
              />
              <Circle
                cx={node.x}
                cy={node.y}
                r={2}
                fill="#0050FF"
              />
            </G>
          ))}
        </G>
      </Svg>
    </View>
  );
};

export const VoiceScreen: React.FC<VoiceScreenProps> = ({ navigation }) => {
  const [isListening, setIsListening] = useState(false);
  
  // Animation values
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const welcomeSlide = useRef(new Animated.Value(-30)).current;
  const orbScale = useRef(new Animated.Value(0.8)).current;
  const subtitleFade = useRef(new Animated.Value(0)).current;
  
  const navigateToDashboard = () => {
    navigation.navigate('MainTabs');
  };

  useEffect(() => {
    // Entrance animations
    Animated.sequence([
      // Welcome text slides down and fades in
      Animated.parallel([
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 800,
          delay: 300,
          useNativeDriver: true,
        }),
        Animated.spring(welcomeSlide, {
          toValue: 0,
          friction: 8,
          tension: 40,
          delay: 300,
          useNativeDriver: true,
        }),
      ]),
      // Orb scales up
      Animated.spring(orbScale, {
        toValue: 1,
        friction: 6,
        tension: 40,
        useNativeDriver: true,
      }),
      // Subtitle fades in
      Animated.timing(subtitleFade, {
        toValue: 1,
        duration: 600,
        useNativeDriver: true,
      }),
    ]).start();
  }, []);

  const handleOrbPress = () => {
    setIsListening(!isListening);
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#000000" />
      
      <LinearGradient
        colors={['#000000', '#000411', '#000824', '#000411', '#000000'] as const}
        style={styles.gradient}
        locations={[0, 0.2, 0.5, 0.8, 1]}
      >
        {/* Neural background */}
        <NeuralBackground />
        
        {/* Main content */}
        <View style={styles.content}>
          {/* Welcome Text - Above the orb */}
          <Animated.View
            style={[
              styles.welcomeContainer,
              {
                opacity: fadeAnim,
                transform: [{ translateY: welcomeSlide }],
              },
            ]}
          >
            <Text style={styles.welcomeText}>Welcome to Electra</Text>
          </Animated.View>

          {/* Main Orb */}
          <Animated.View
            style={[
              styles.orbContainer,
              {
                opacity: fadeAnim,
                transform: [{ scale: orbScale }],
              },
            ]}
          >
            <TouchableOpacity onPress={handleOrbPress} activeOpacity={0.9}>
              <ElectraOrb isListening={isListening} />
            </TouchableOpacity>
          </Animated.View>

          {/* Subtitle */}
          <Animated.View
            style={[
              styles.subtitleContainer,
              {
                opacity: subtitleFade,
              },
            ]}
          >
            {isListening ? (
              <Text style={styles.listeningText}>Listening...</Text>
            ) : (
              <Text style={styles.speakText}>Speak or Ask Anything</Text>
            )}
          </Animated.View>

          {/* Swipe hint */}
          <Animated.View
            style={[
              styles.swipeContainer,
              {
                opacity: subtitleFade,
              },
            ]}
          >
            <TouchableOpacity onPress={navigateToDashboard} style={styles.swipeButton}>
              <Text style={styles.swipeText}>Swipe to access interactive dashboard</Text>
              <View style={styles.swipeIndicator}>
                <View style={styles.swipeLine} />
              </View>
            </TouchableOpacity>
          </Animated.View>
        </View>
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
  content: {
    flex: 1,
    alignItems: 'center',
    paddingTop: screenHeight * 0.12,
  },
  welcomeContainer: {
    marginBottom: 40,
  },
  welcomeText: {
    fontSize: 42,
    fontWeight: '200',
    color: '#FFFFFF',
    letterSpacing: 3,
    textAlign: 'center',
    textShadowColor: '#0050FF',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 20,
  },
  orbContainer: {
    marginBottom: 30,
    shadowColor: '#0050FF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 30,
  },
  subtitleContainer: {
    marginBottom: 60,
  },
  speakText: {
    fontSize: 20,
    fontWeight: '300',
    color: '#A6B0C3',
    letterSpacing: 1.5,
    textAlign: 'center',
  },
  listeningText: {
    fontSize: 22,
    fontWeight: '400',
    color: '#00D4FF',
    letterSpacing: 2,
    textAlign: 'center',
    textShadowColor: '#00D4FF',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  swipeContainer: {
    position: 'absolute',
    bottom: 50,
    alignItems: 'center',
  },
  swipeButton: {
    alignItems: 'center',
    padding: 10,
  },
  swipeText: {
    fontSize: 14,
    color: '#4A5568',
    marginBottom: 12,
    letterSpacing: 0.5,
  },
  swipeIndicator: {
    width: 40,
    height: 3,
    backgroundColor: 'rgba(74, 85, 104, 0.3)',
    borderRadius: 1.5,
    overflow: 'hidden',
  },
  swipeLine: {
    width: '100%',
    height: '100%',
    backgroundColor: 'rgba(74, 85, 104, 0.6)',
    borderRadius: 1.5,
  },
});