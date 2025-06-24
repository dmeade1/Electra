// src/components/common/ElectraOrb.tsx
import React, { useEffect, useRef, useMemo } from 'react';
import {
  View,
  StyleSheet,
  Animated,
  ViewStyle,
  Dimensions,
  Easing,
  Platform,
} from 'react-native';
import Svg, {
  Circle,
  G,
  Path,
  Defs,
  RadialGradient,
  Stop,
  ClipPath,
} from 'react-native-svg';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const { width: screenWidth } = Dimensions.get('window');

interface ElectraOrbProps {
  size?: number;
  isListening?: boolean;
  style?: ViewStyle;
  quality?: 'low' | 'medium' | 'high' | 'ultra';
}

interface Point {
  x: number;
  y: number;
}

// Pre-calculate common values
const NEURAL_CONNECTIONS_CACHE = new Map<string, string>();

export const ElectraOrb: React.FC<ElectraOrbProps> = ({
  size = Math.min(screenWidth * 0.85, 340),
  isListening = false,
  style,
  quality = 'medium',
}) => {
  const centerX = size / 2;
  const centerY = size / 2;
  const radius = size * 0.48;

  // Animation refs
  const pulseAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;

  // Performance settings based on quality
  const qualitySettings = useMemo(() => ({
    low: {
      tendrils: 6,
      tendrilSegments: 8,
      forkProbability: 0,
      neuralNodes: 0,
      nodes: 15,
      enableGlow: false,
      enableNeuralWeb: false,
      enablePulse: false,
    },
    medium: {
      tendrils: 8,
      tendrilSegments: 12,
      forkProbability: 0.3,
      neuralNodes: 150,
      nodes: 30,
      enableGlow: false,
      enableNeuralWeb: true,
      enablePulse: true,
    },
    high: {
      tendrils: 12,
      tendrilSegments: 16,
      forkProbability: 0.5,
      neuralNodes: 300,
      nodes: 50,
      enableGlow: true,
      enableNeuralWeb: true,
      enablePulse: true,
    },
    ultra: {
      tendrils: 14,
      tendrilSegments: 20,
      forkProbability: 0.6,
      neuralNodes: 400,
      nodes: 80,
      enableGlow: true,
      enableNeuralWeb: true,
      enablePulse: true,
    },
  }), []);

  const settings = qualitySettings[quality];

  // Optimized structure generation
  const orbData = useMemo(() => {
    const cacheKey = `${size}-${quality}`;
    
    // Lightning tendrils
    const tendrils: string[] = [];
    const nodes: Point[] = [];
    
    // Generate main tendrils
    for (let i = 0; i < settings.tendrils; i++) {
      const angle = (Math.PI * 2 / settings.tendrils) * i;
      const length = radius * 0.85;
      
      // Build path with fewer segments
      const pathSegments: string[] = [`M ${centerX} ${centerY}`];
      
      for (let j = 1; j <= settings.tendrilSegments; j++) {
        const progress = j / settings.tendrilSegments;
        const dist = length * progress;
        
        // Simplified wiggle
        const wiggle = Math.sin(progress * Math.PI * 2.5) * 0.08 * (1 - progress);
        const currentAngle = angle + wiggle;
        
        const x = centerX + Math.cos(currentAngle) * dist;
        const y = centerY + Math.sin(currentAngle) * dist;
        
        pathSegments.push(`L ${x.toFixed(1)} ${y.toFixed(1)}`);
        
        // Add end node
        if (j === settings.tendrilSegments) {
          nodes.push({ x, y });
        }
        
        // Simplified forks
        if (settings.forkProbability > 0 && 
            j === Math.floor(settings.tendrilSegments * 0.6) && 
            Math.random() < settings.forkProbability) {
          const forkAngle = currentAngle + (Math.random() - 0.5) * 0.5;
          const forkLength = length * 0.3;
          const forkX = x + Math.cos(forkAngle) * forkLength;
          const forkY = y + Math.sin(forkAngle) * forkLength;
          
          tendrils.push(`M ${x.toFixed(1)} ${y.toFixed(1)} L ${forkX.toFixed(1)} ${forkY.toFixed(1)}`);
          nodes.push({ x: forkX, y: forkY });
        }
      }
      
      tendrils.push(pathSegments.join(' '));
    }
    
    // Neural web - organic curved connections
    let neuralWebPath = '';
    if (settings.enableNeuralWeb && settings.neuralNodes > 0) {
      const cached = NEURAL_CONNECTIONS_CACHE.get(cacheKey);
      if (cached) {
        neuralWebPath = cached;
      } else {
        const webSegments: string[] = [];
        const neuralNodesList: Point[] = [];
        
        // Generate neural nodes in organic clusters
        const clusters = 8;
        const nodesPerCluster = Math.floor(settings.neuralNodes / clusters);
        
        for (let c = 0; c < clusters; c++) {
          const clusterAngle = (Math.PI * 2 / clusters) * c;
          const clusterRadius = radius * (0.3 + Math.random() * 0.4);
          const clusterX = centerX + Math.cos(clusterAngle) * clusterRadius;
          const clusterY = centerY + Math.sin(clusterAngle) * clusterRadius;
          
          for (let n = 0; n < nodesPerCluster; n++) {
            const nodeAngle = Math.random() * Math.PI * 2;
            const nodeRadius = Math.random() * radius * 0.15;
            const x = clusterX + Math.cos(nodeAngle) * nodeRadius;
            const y = clusterY + Math.sin(nodeAngle) * nodeRadius;
            
            // Check if node is within orb bounds
            const distFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
            if (distFromCenter <= radius * 0.95) {
              neuralNodesList.push({ x, y });
            }
          }
        }
        
        // Create organic connections between nearby nodes
        neuralNodesList.forEach((node, i) => {
          // Find 2-4 nearest neighbors
          const distances = neuralNodesList
            .map((other, j) => ({
              index: j,
              dist: Math.sqrt((node.x - other.x) ** 2 + (node.y - other.y) ** 2)
            }))
            .filter(d => d.index !== i && d.dist < radius * 0.25 && d.dist > radius * 0.02)
            .sort((a, b) => a.dist - b.dist)
            .slice(0, 2 + Math.floor(Math.random() * 2));
          
          distances.forEach(({ index }) => {
            if (index > i) { // Avoid duplicate connections
              const other = neuralNodesList[index];
              
              // Create curved path using quadratic bezier
              const midX = (node.x + other.x) / 2;
              const midY = (node.y + other.y) / 2;
              
              // Add organic curve by offsetting the control point
              const curvature = 0.2;
              const perpX = -(other.y - node.y) * curvature;
              const perpY = (other.x - node.x) * curvature;
              
              const controlX = midX + perpX * (Math.random() - 0.5) * 2;
              const controlY = midY + perpY * (Math.random() - 0.5) * 2;
              
              webSegments.push(
                `M ${node.x.toFixed(1)} ${node.y.toFixed(1)} Q ${controlX.toFixed(1)} ${controlY.toFixed(1)} ${other.x.toFixed(1)} ${other.y.toFixed(1)}`
              );
            }
          });
        });
        
        neuralWebPath = webSegments.join(' ');
        NEURAL_CONNECTIONS_CACHE.set(cacheKey, neuralWebPath);
      }
    }
    
    // Distributed nodes
    const innerRadius = radius * 0.4;
    const outerRadius = radius * 0.8;
    for (let i = 0; i < settings.nodes; i++) {
      const angle = (Math.PI * 2 / settings.nodes) * i;
      const r = innerRadius + (outerRadius - innerRadius) * ((i % 3) / 2);
      nodes.push({
        x: centerX + Math.cos(angle) * r,
        y: centerY + Math.sin(angle) * r,
      });
    }
    
    // Center node
    nodes.push({ x: centerX, y: centerY });
    
    return {
      tendrils: tendrils.join(' '),
      nodes,
      neuralWebPath,
    };
  }, [size, centerX, centerY, radius, settings, qualitySettings]);

  // Optimized pulse animation
  useEffect(() => {
    if (!settings.enablePulse) return;
    
    const animation = Animated.loop(
      Animated.timing(pulseAnim, {
        toValue: 1,
        duration: 4000,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      })
    );
    animation.start();
    return () => animation.stop();
  }, [pulseAnim, settings.enablePulse]);

  // Voice activation
  useEffect(() => {
    Animated.spring(scaleAnim, {
      toValue: isListening ? 1.05 : 1,
      friction: 8,
      tension: 40,
      useNativeDriver: true,
    }).start();
  }, [isListening, scaleAnim]);

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
        <View style={[styles.orb, { width: size, height: size }]}>
          <Svg 
            width={size} 
            height={size} 
            viewBox={`0 0 ${size} ${size}`}
            renderToHardwareTextureAndroid={Platform.OS === 'android'}
            shouldRasterizeIOS={Platform.OS === 'ios'}
          >
            <Defs>
              {/* Simplified gradients */}
              <RadialGradient id="orb" cx="50%" cy="50%">
                <Stop offset="0%" stopColor="#04a3f5" stopOpacity="1" />
                <Stop offset="50%" stopColor="#007bff" stopOpacity="0.7" />
                <Stop offset="100%" stopColor="#001830" stopOpacity="0" />
              </RadialGradient>

              <RadialGradient id="center" cx="50%" cy="50%">
                <Stop offset="0%" stopColor="#00faff" stopOpacity="1" />
                <Stop offset="100%" stopColor="#007bff" stopOpacity="0" />
              </RadialGradient>

              <ClipPath id="clip">
                <Circle cx={centerX} cy={centerY} r={radius} />
              </ClipPath>
            </Defs>

            {/* Black background */}
            <Circle cx={centerX} cy={centerY} r={radius} fill="#000000" />

            <G clipPath="url(#clip)">
              {/* Main orb */}
              <Circle cx={centerX} cy={centerY} r={radius} fill="url(#orb)" />

              {/* Neural web with organic curves */}
              {settings.enableNeuralWeb && orbData.neuralWebPath && (
                <G opacity={0.6}>
                  <Path
                    d={orbData.neuralWebPath}
                    stroke="rgba(0,123,255,0.15)"
                    strokeWidth={0.5}
                    fill="none"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </G>
              )}

              {/* All tendrils as single path */}
              <Path
                d={orbData.tendrils}
                stroke="#007bff"
                strokeWidth={1.5}
                fill="none"
                opacity={0.7}
                strokeLinecap="round"
              />

              {/* Optimized nodes */}
              {orbData.nodes.map((node, index) => {
                const isCenter = index === orbData.nodes.length - 1;
                return (
                  <Circle
                    key={index}
                    cx={node.x}
                    cy={node.y}
                    r={isCenter ? 6 : 2}
                    fill={isCenter ? "#04a3f5" : "#007bff"}
                    opacity={isCenter ? 1 : 0.8}
                  />
                );
              })}

              {/* Animated center pulse */}
              {settings.enablePulse && (
                <AnimatedCircle
                  cx={centerX}
                  cy={centerY}
                  r={radius * 0.3}
                  fill="url(#center)"
                  opacity={pulseAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0.3, 0.6],
                  })}
                />
              )}
            </G>

            {/* Simple edge */}
            <Circle
              cx={centerX}
              cy={centerY}
              r={radius}
              stroke="rgba(0,123,255,0.3)"
              strokeWidth={1}
              fill="none"
            />
          </Svg>
        </View>
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
});