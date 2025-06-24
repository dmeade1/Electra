// src/components/common/ElectraOrb.tsx
import React, { useEffect, useRef } from 'react';
import {
  View,
  StyleSheet,
  Animated,
  ViewStyle,
} from 'react-native';
import Svg, { Circle, G, Path, Defs, RadialGradient, Stop } from 'react-native-svg';
import { theme } from '../../styles/theme';

interface ElectraOrbProps {
  size?: number;
  isListening?: boolean;
  isPulsing?: boolean;
  style?: ViewStyle;
}

export const ElectraOrb: React.FC<ElectraOrbProps> = ({
  size = 200,
  isListening = false,
  isPulsing = true,
  style,
}) => {
  const rotateAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;
  const glowAnim = useRef(new Animated.Value(0.5)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    // Rotation animation
    Animated.loop(
      Animated.timing(rotateAnim, {
        toValue: 1,
        duration: 20000,
        useNativeDriver: true,
      })
    ).start();

    // Glow animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, {
          toValue: 0.8,
          duration: 2000,
          useNativeDriver: false,
        }),
        Animated.timing(glowAnim, {
          toValue: 0.5,
          duration: 2000,
          useNativeDriver: false,
        }),
      ])
    ).start();
  }, []);

  // Voice activation animation
  useEffect(() => {
    if (isListening) {
      Animated.spring(scaleAnim, {
        toValue: 1.2,
        useNativeDriver: true,
      }).start();
    } else {
      Animated.spring(scaleAnim, {
        toValue: 1,
        useNativeDriver: true,
      }).start();
    }
  }, [isListening]);

  const spin = rotateAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <View style={[styles.container, { width: size, height: size }, style]}>
      <Animated.View
        style={[
          styles.orbContainer,
          {
            transform: [{ scale: scaleAnim }],
          },
        ]}
      >
        {/* Outer glow effect */}
        <Animated.View
          style={[
            styles.outerGlow,
            {
              width: size * 1.4,
              height: size * 1.4,
              opacity: glowAnim,
            },
          ]}
        />

        {/* Main orb */}
        <Animated.View
          style={[
            styles.orb,
            {
              width: size,
              height: size,
              transform: [{ rotate: spin }],
            },
          ]}
        >
          <Svg width={size} height={size} viewBox="0 0 200 200">
            <Defs>
              <RadialGradient id="orbGradient" cx="50%" cy="50%">
                <Stop offset="0%" stopColor={theme.colors.primary.light} stopOpacity="0.8" />
                <Stop offset="50%" stopColor={theme.colors.primary.main} stopOpacity="0.6" />
                <Stop offset="100%" stopColor={theme.colors.primary.dark} stopOpacity="0.3" />
              </RadialGradient>
            </Defs>

            <Circle
              cx="100"
              cy="100"
              r="95"
              fill="url(#orbGradient)"
            />

            {/* Lightning effect */}
            <G opacity={0.8}>
              <Path
                d="M100,50 L90,80 L110,80 L100,150"
                stroke={theme.colors.primary.light}
                strokeWidth="3"
                fill="none"
                strokeLinecap="round"
              />
            </G>

            <Circle
              cx="100"
              cy="100"
              r="95"
              stroke={theme.colors.primary.light}
              strokeWidth="1"
              fill="none"
              opacity={0.8}
            />
          </Svg>
        </Animated.View>
      </Animated.View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  orbContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  orb: {
    position: 'absolute',
  },
  outerGlow: {
    position: 'absolute',
    backgroundColor: theme.colors.primary.glow,
    borderRadius: 999,
  },
});