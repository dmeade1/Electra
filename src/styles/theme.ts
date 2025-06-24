// src/styles/theme.ts
export const theme = {
  colors: {
    // Primary electric blue palette
    primary: {
      main: '#00D4FF',
      light: '#4DE3FF',
      dark: '#0099CC',
      glow: 'rgba(0, 212, 255, 0.6)',
      subtle: 'rgba(0, 212, 255, 0.1)',
      neural: '#00E5FF', // For synapses
    },
    
    // Pure black theme
    background: {
      primary: '#000000',
      secondary: '#0A0A0A',
      tertiary: '#111111',
      card: '#1A1A1A',
      overlay: 'rgba(0, 0, 0, 0.95)',
    },
    
    // Text colors
    text: {
      primary: '#FFFFFF',
      secondary: '#A6B0C3',
      tertiary: '#6B7280',
      disabled: '#4B5563',
    },
    
    // Trading colors - ADD THIS SECTION
    trading: {
      profit: '#00FF88',
      loss: '#FF3366',
      neutral: '#FFD700',
      bull: '#00C851',
      bear: '#FF3547',
    },
    
    // Status colors
    success: '#00FF88',
    warning: '#FFD700',
    error: '#FF3366',
    info: '#00D4FF',
    
    // Additional accent colors
    accent: {
      purple: '#8B5CF6',
      pink: '#EC4899',
      cyan: '#06B6D4',
    },
  },
  
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
    xxxl: 64,
  },
  
  borderRadius: {
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    full: 9999,
  },
  
  typography: {
    fontSize: {
      xs: 12,
      sm: 14,
      md: 16,
      lg: 18,
      xl: 24,
      xxl: 32,
      xxxl: 48,
      giant: 64,
    },
  },
  
  shadows: {
    sm: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.1,
      shadowRadius: 3,
      elevation: 3,
    },
    md: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 0.15,
      shadowRadius: 6,
      elevation: 6,
    },
    lg: {
      shadowColor: '#000',
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 0.2,
      shadowRadius: 12,
      elevation: 12,
    },
    glow: {
      shadowColor: '#00D4FF',
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.5,
      shadowRadius: 20,
      elevation: 0,
    },
    neuralGlow: {
      shadowColor: '#00E5FF',
      shadowOffset: { width: 0, height: 0 },
      shadowOpacity: 0.8,
      shadowRadius: 30,
      elevation: 0,
    },
  },
  
  animation: {
    duration: {
      fast: 200,
      normal: 300,
      slow: 500,
    },
    easing: {
      easeIn: 'ease-in',
      easeOut: 'ease-out',
      easeInOut: 'ease-in-out',
    },
  },
};