// src/components/common/ElectraOrb.tsx
import React, { useEffect, useRef, useMemo } from 'react';
import {
  View,
  StyleSheet,
  Animated,
  ViewStyle,
  Dimensions,
  Easing,
} from 'react-native';
import Svg, {
  Circle,
  G,
  Path,
  Defs,
  RadialGradient,
  Stop,
  ClipPath,
  Filter,
  FeGaussianBlur,
} from 'react-native-svg';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

const { width: screenWidth } = Dimensions.get('window');

interface ElectraOrbProps {
  size?: number;
  isListening?: boolean;
  style?: ViewStyle;
}

interface Point {
  x: number;
  y: number;
}

interface Branch {
  path: string;
  nodes: Point[];
  width?: number;
  opacity?: number;
}

interface NeuralConnection {
  from: Point;
  to: Point;
  opacity: number;
  path?: string;
}

interface Node {
  x: number;
  y: number;
  size: number;
  glowSize: number;
}

export const ElectraOrb: React.FC<ElectraOrbProps> = ({
  size = Math.min(screenWidth * 0.85, 340),
  isListening = false,
  style,
}) => {
  const centerX = size / 2;
  const centerY = size / 2;
  // 90% of half-width as per description (keeping original's 48% for now as it works well visually)
  const radius = size * 0.48;

  // Animation refs
  const pulseAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(1)).current;

  // Generate lightning pattern and neural map
  const { branches, nodes, neuralMap } = useMemo(() => {
    const branchList: Branch[] = [];
    const nodesList: Node[] = [];
    const neuralNodes: Point[] = [];
    const neuralConnections: NeuralConnection[] = [];
    
    // Create background neural map grid - extends to edges (from original)
    const gridSize = 24;
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const x = (i / (gridSize - 1)) * radius * 2.2 - radius * 1.1 + centerX;
        const y = (j / (gridSize - 1)) * radius * 2.2 - radius * 1.1 + centerY;
        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        
        if (dist <= radius * 1.05) {
          const offsetX = (Math.random() - 0.5) * 12;
          const offsetY = (Math.random() - 0.5) * 12;
          const finalX = x + offsetX;
          const finalY = y + offsetY;
          const finalDist = Math.sqrt((finalX - centerX) ** 2 + (finalY - centerY) ** 2);
          
          if (finalDist <= radius) {
            neuralNodes.push({ x: finalX, y: finalY });
          }
        }
      }
    }
    
    // Add extra nodes near the edge (from original)
    for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 20) {
      for (let r = radius * 0.85; r <= radius; r += radius * 0.05) {
        const x = centerX + Math.cos(angle) * r + (Math.random() - 0.5) * 10;
        const y = centerY + Math.sin(angle) * r + (Math.random() - 0.5) * 10;
        const dist = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        if (dist <= radius) {
          neuralNodes.push({ x, y });
        }
      }
    }
    
    // Connect nearby neural nodes (enhanced from original with curved paths)
    neuralNodes.forEach((node, i) => {
      neuralNodes.forEach((other, j) => {
        if (i < j) {
          const dist = Math.sqrt((node.x - other.x) ** 2 + (node.y - other.y) ** 2);
          if (dist < radius * 0.15 && dist > radius * 0.01) {
            const nodeDistFromCenter = Math.sqrt((node.x - centerX) ** 2 + (node.y - centerY) ** 2);
            const otherDistFromCenter = Math.sqrt((other.x - centerX) ** 2 + (other.y - centerY) ** 2);
            const avgDistFromCenter = (nodeDistFromCenter + otherDistFromCenter) / 2;
            
            // Higher opacity near edges, lower near center
            const opacity = 0.1 + (avgDistFromCenter / radius) * 0.1;
            
            // Create curved path for web connections
            const midX = (node.x + other.x) / 2 + (Math.random() - 0.5) * 5;
            const midY = (node.y + other.y) / 2 + (Math.random() - 0.5) * 5;
            const path = `M ${node.x} ${node.y} Q ${midX} ${midY} ${other.x} ${other.y}`;
            
            neuralConnections.push({ 
              from: node, 
              to: other,
              opacity: opacity * (0.5 + Math.random() * 0.5),
              path
            });
          }
        }
      });
    });
    
    // Create 12-14 main lightning tendrils (as per description)
    const tendrilCount = 12 + Math.floor(Math.random() * 3);
    
    const createLightningPath = (startAngle: number, variance: number = 0.3) => {
      const points: Point[] = [];
      let path = `M ${centerX} ${centerY}`;
      
      const length = radius * (0.85 + Math.random() * 0.1);
      const segments = 20;
      
      let angle = startAngle;
      let lastX = centerX;
      let lastY = centerY;
      
      // Create main branch with sinusoidal wiggle (from description)
      for (let i = 1; i <= segments; i++) {
        const progress = i / segments;
        const dist = length * progress;
        
        // Sinusoidal wiggle with decreasing amplitude
        const wiggleAmplitude = 0.15 * (1 - progress);
        angle = startAngle + Math.sin(progress * Math.PI * 3) * wiggleAmplitude;
        
        const x = centerX + Math.cos(angle) * dist;
        const y = centerY + Math.sin(angle) * dist;
        
        // Smooth bezier curve
        const cp1x = lastX + (x - lastX) * 0.3;
        const cp1y = lastY + (y - lastY) * 0.3;
        const cp2x = x - (x - lastX) * 0.3;
        const cp2y = y - (y - lastY) * 0.3;
        
        path += ` C ${cp1x} ${cp1y} ${cp2x} ${cp2y} ${x} ${y}`;
        
        points.push({ x, y });
        lastX = x;
        lastY = y;
        
        // Add node at tendril end
        if (i === segments) {
          nodesList.push({
            x,
            y,
            size: 1.5 + Math.random() * 1,
            glowSize: 10 + Math.random() * 5,
          });
        }
        
        // Fork branches (1-2 per tendril as per description)
        if (i > segments * 0.4 && i < segments * 0.8 && Math.random() > 0.6) {
          const subAngle = angle + (Math.random() - 0.5) * 0.8;
          const subLength = (length - dist) * (0.4 + Math.random() * 0.3);
          const subSegments = 8;
          
          let subPath = `M ${x} ${y}`;
          let subLastX = x;
          let subLastY = y;
          let currentAngle = subAngle;
          
          for (let j = 1; j <= subSegments; j++) {
            const subProgress = j / subSegments;
            const subDist = subLength * subProgress;
            
            currentAngle += (Math.random() - 0.5) * 0.4;
            
            const subX = x + Math.cos(currentAngle) * subDist;
            const subY = y + Math.sin(currentAngle) * subDist;
            
            const scp1x = subLastX + (subX - subLastX) * 0.3;
            const scp1y = subLastY + (subY - subLastY) * 0.3;
            const scp2x = subX - (subX - subLastX) * 0.3;
            const scp2y = subY - (subY - subLastY) * 0.3;
            
            subPath += ` C ${scp1x} ${scp1y} ${scp2x} ${scp2y} ${subX} ${subY}`;
            
            if (j === subSegments) {
              nodesList.push({
                x: subX,
                y: subY,
                size: 1.2 + Math.random() * 0.8,
                glowSize: 8 + Math.random() * 4,
              });
            }
            
            subLastX = subX;
            subLastY = subY;
          }
          
          branchList.push({ 
            path: subPath, 
            nodes: [],
            width: 0.8 + Math.random() * 0.4,
            opacity: 0.5 + Math.random() * 0.2
          });
        }
      }
      
      return { 
        path, 
        nodes: points,
        width: 1.5 + Math.random() * 0.5,
        opacity: 0.6 + Math.random() * 0.1
      };
    };
    
    // Create main branches with equal angular spacing
    for (let i = 0; i < tendrilCount; i++) {
      const angle = (Math.PI * 2 / tendrilCount) * i;
      const branch = createLightningPath(angle);
      branchList.push(branch);
    }
    
    // Add nodes in density ring (40-80% of radius as per description)
    const innerRadius = radius * 0.4;
    const outerRadius = radius * 0.8;
    for (let i = 0; i < 50; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = innerRadius + Math.random() * (outerRadius - innerRadius);
      nodesList.push({
        x: centerX + Math.cos(angle) * r,
        y: centerY + Math.sin(angle) * r,
        size: 1.5 + Math.random() * 1,
        glowSize: 6 + Math.random() * 9,
      });
    }
    
    // Add central bright node
    nodesList.push({
      x: centerX,
      y: centerY,
      size: 8,
      glowSize: 50,
    });
    
    return { 
      branches: branchList, 
      nodes: nodesList, 
      neuralMap: { nodes: neuralNodes, connections: neuralConnections } 
    };
  }, [size, centerX, centerY, radius]);

  // Gentle pulse animation
  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 3000,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: false,
        }),
        Animated.timing(pulseAnim, {
          toValue: 0,
          duration: 3000,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: false,
        }),
      ])
    );
    animation.start();
    return () => animation.stop();
  }, [pulseAnim]);

  // Voice activation scale
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
          <Svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            <Defs>
              {/* Main orb gradient - exact colors from description */}
              <RadialGradient id="orbGradient" cx="50%" cy="50%">
                <Stop offset="0%" stopColor="#04a3f5" stopOpacity="1" />
                <Stop offset="30%" stopColor="#007bff" stopOpacity="0.9" />
                <Stop offset="70%" stopColor="#007bff" stopOpacity="0.5" />
                <Stop offset="100%" stopColor="rgba(0,40,90,0.1)" stopOpacity="0" />
              </RadialGradient>

              {/* Center core gradient */}
              <RadialGradient id="centerCore" cx="50%" cy="50%">
                <Stop offset="0%" stopColor="#04a3f5" stopOpacity="1" />
                <Stop offset="40%" stopColor="#00faff" stopOpacity="0.8" />
                <Stop offset="100%" stopColor="#007bff" stopOpacity="0" />
              </RadialGradient>

              {/* Node glow gradient */}
              <RadialGradient id="nodeGlow" cx="50%" cy="50%">
                <Stop offset="0%" stopColor="#007bff" stopOpacity="0.9" />
                <Stop offset="50%" stopColor="#007bff" stopOpacity="0.5" />
                <Stop offset="100%" stopColor="#007bff" stopOpacity="0" />
              </RadialGradient>

              {/* Tendril glow filter */}
              <Filter id="tendrilGlow">
                <FeGaussianBlur stdDeviation="3" />
              </Filter>

              <ClipPath id="orbClip">
                <Circle cx={centerX} cy={centerY} r={radius} />
              </ClipPath>
            </Defs>

            {/* Pure black background (#000000) */}
            <Circle cx={centerX} cy={centerY} r={size / 2} fill="#000000" />

            <G clipPath="url(#orbClip)">
              {/* Main orb with proper gradient */}
              <Circle cx={centerX} cy={centerY} r={radius} fill="url(#orbGradient)" />

              {/* Inner aura bloom */}
              <AnimatedCircle
                cx={centerX}
                cy={centerY}
                r={100}
                fill="rgba(0,255,255,0.05)"
                opacity={pulseAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0.8, 1],
                })}
              />

              {/* Neural network with curved paths */}
              <G>
                {neuralMap.connections.map((conn, index) => {
                  const path = conn.path || `M ${conn.from.x} ${conn.from.y} L ${conn.to.x} ${conn.to.y}`;
                  return (
                    <Path
                      key={`neural-${index}`}
                      d={path}
                      stroke="rgba(0,123,255,0.15)"
                      strokeWidth={0.4}
                      opacity={conn.opacity}
                      fill="none"
                    />
                  );
                })}
              </G>

              {/* Lightning branches with proper glow */}
              <G>
                {branches.map((branch, index) => (
                  <G key={`branch-${index}`}>
                    {/* Shadow blur glow */}
                    <Path
                      d={branch.path}
                      stroke="rgba(0,123,255,0.3)"
                      strokeWidth={(branch.width || 1.5) * 8}
                      fill="none"
                      opacity={(branch.opacity || 0.7) * 0.3}
                      strokeLinecap="round"
                      filter="url(#tendrilGlow)"
                    />
                    {/* Main tendril */}
                    <Path
                      d={branch.path}
                      stroke="rgba(0,123,255,0.7)"
                      strokeWidth={branch.width || 1.5}
                      fill="none"
                      opacity={branch.opacity || 0.7}
                      strokeLinecap="round"
                    />
                  </G>
                ))}
              </G>

              {/* Glowing nodes */}
              <G>
                {nodes.map((node, index) => {
                  const isCenter = node.x === centerX && node.y === centerY;
                  
                  return (
                    <G key={`node-${index}`}>
                      {/* Node glow */}
                      <Circle
                        cx={node.x}
                        cy={node.y}
                        r={node.glowSize}
                        fill="url(#nodeGlow)"
                        opacity={0.5}
                      />
                      {/* Node core */}
                      <Circle
                        cx={node.x}
                        cy={node.y}
                        r={node.size}
                        fill={isCenter ? "#04a3f5" : "rgba(0,123,255,0.9)"}
                        opacity={1}
                      />
                    </G>
                  );
                })}
              </G>

              {/* Central bright core (8px radius as per description) */}
              <G>
                <AnimatedCircle
                  cx={centerX}
                  cy={centerY}
                  r={50}
                  fill="url(#centerCore)"
                  opacity={pulseAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [0.6, 0.8],
                  })}
                />
                <Circle
                  cx={centerX}
                  cy={centerY}
                  r={8}
                  fill="#04a3f5"
                  opacity={1}
                />
              </G>
            </G>

            {/* Edge highlight with proper feathering */}
            <Circle
              cx={centerX}
              cy={centerY}
              r={radius}
              stroke="rgba(0,123,255,0.4)"
              strokeWidth={2}
              fill="none"
              opacity={0.8}
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