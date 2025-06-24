// src/components/common/GlassCard.tsx
import React from 'react';
import { View, StyleSheet, ViewStyle } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { theme } from '../../styles/theme';

interface GlassCardProps {
  children: React.ReactNode;
  style?: ViewStyle;
  variant?: 'default' | 'primary' | 'dark';
}

export const GlassCard: React.FC<GlassCardProps> = ({
  children,
  style,
  variant = 'default',
}) => {
  const getColors = (): readonly [string, string, ...string[]] => {  // ← Fixed type
    switch (variant) {
      case 'primary':
        return ['rgba(0, 212, 255, 0.1)', 'rgba(0, 212, 255, 0.05)'] as const;
      case 'dark':
        return ['rgba(0, 0, 0, 0.3)', 'rgba(0, 0, 0, 0.1)'] as const;
      default:
        return ['rgba(255, 255, 255, 0.08)', 'rgba(255, 255, 255, 0.03)'] as const;
    }
  };

  return (
    <View style={[styles.container, style]}>
      <LinearGradient
        colors={getColors()}
        style={styles.gradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        <View style={styles.content}>{children}</View>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    borderRadius: 16,
    overflow: 'hidden',
  },
  gradient: {
    borderRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  content: {
    padding: 24,
  },
});