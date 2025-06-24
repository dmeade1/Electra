// src/components/voice/VoiceWaveform.tsx
import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated } from 'react-native';
import { theme } from '../../styles/theme';

interface VoiceWaveformProps {
  isActive: boolean;
  amplitude?: number;
}

export const VoiceWaveform: React.FC<VoiceWaveformProps> = ({
  isActive,
  amplitude = 0.5,
}) => {
  const bars = useRef(
    Array(5)
      .fill(0)
      .map(() => new Animated.Value(0.3))
  ).current;

  useEffect(() => {
    if (isActive) {
      bars.forEach((bar, index) => {
        Animated.loop(
          Animated.sequence([
            Animated.timing(bar, {
              toValue: Math.random() * amplitude + 0.5,
              duration: 200 + index * 50,
              useNativeDriver: true,
            }),
            Animated.timing(bar, {
              toValue: 0.3,
              duration: 200 + index * 50,
              useNativeDriver: true,
            }),
          ])
        ).start();
      });
    } else {
      bars.forEach((bar) => {
        Animated.timing(bar, {
          toValue: 0.3,
          duration: 200,
          useNativeDriver: true,
        }).start();
      });
    }
  }, [isActive, amplitude]);

  return (
    <View style={styles.container}>
      {bars.map((bar, index) => (
        <Animated.View
          key={index}
          style={[
            styles.bar,
            {
              transform: [{ scaleY: bar }],
              opacity: isActive ? 1 : 0.3,
            },
          ]}
        />
      ))}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    height: 60,
  },
  bar: {
    width: 4,
    height: 40,
    backgroundColor: '#00D4FF',
    marginHorizontal: 3,
    borderRadius: 2,
  },
});