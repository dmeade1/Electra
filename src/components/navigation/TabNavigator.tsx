// src/navigation/TabNavigator.tsx
import React from 'react';
import { View, StyleSheet } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { HomeScreen } from 'src/screens/HomeScreen';
import { MarketsScreen } from 'src/screens/MarketsScreen';  // Changed from AnalyticsScreen
import { TradeScreen } from 'src/screens/TradeScreen';      // Changed from CardsScreen
import { ProfileScreen } from 'src/screens/ProfileScreen';

const Tab = createBottomTabNavigator();

export const TabNavigator = () => {
  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          position: 'absolute',
          backgroundColor: 'rgba(0, 0, 0, 0.95)',
          borderTopWidth: 1,
          borderTopColor: 'rgba(255, 255, 255, 0.1)',
          height: 90,
          paddingBottom: 20,
          paddingTop: 10,
        },
        tabBarActiveTintColor: '#00D4FF',
        tabBarInactiveTintColor: '#6B7280',
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
          marginTop: 4,
        },
      }}
    >
      <Tab.Screen
        name="Portfolio"
        component={HomeScreen}
        options={{
          tabBarLabel: 'Portfolio',
          tabBarIcon: ({ color, size }) => (
            <View style={[styles.iconContainer, { backgroundColor: color === '#00D4FF' ? 'rgba(0, 212, 255, 0.1)' : 'transparent' }]}>
              <PortfolioIcon color={color} size={size} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Markets"
        component={MarketsScreen}
        options={{
          tabBarLabel: 'Markets',
          tabBarIcon: ({ color, size }) => (
            <View style={[styles.iconContainer, { backgroundColor: color === '#00D4FF' ? 'rgba(0, 212, 255, 0.1)' : 'transparent' }]}>
              <MarketsIcon color={color} size={size} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Trade"
        component={TradeScreen}
        options={{
          tabBarLabel: 'Trade',
          tabBarIcon: ({ color, size }) => (
            <View style={[styles.iconContainer, { backgroundColor: color === '#00D4FF' ? 'rgba(0, 212, 255, 0.1)' : 'transparent' }]}>
              <TradeIcon color={color} size={size} />
            </View>
          ),
        }}
      />
      <Tab.Screen
        name="Profile"
        component={ProfileScreen}
        options={{
          tabBarLabel: 'Profile',
          tabBarIcon: ({ color, size }) => (
            <View style={[styles.iconContainer, { backgroundColor: color === '#00D4FF' ? 'rgba(0, 212, 255, 0.1)' : 'transparent' }]}>
              <ProfileIcon color={color} size={size} />
            </View>
          ),
        }}
      />
    </Tab.Navigator>
  );
};

// Updated icons for trading
const PortfolioIcon = ({ color, size }: { color: string; size: number }) => (
  <View style={[styles.icon, { width: size, height: size }]}>
    <View style={styles.portfolioBars}>
      <View style={[styles.bar, { height: '60%', backgroundColor: color }]} />
      <View style={[styles.bar, { height: '80%', backgroundColor: color }]} />
      <View style={[styles.bar, { height: '40%', backgroundColor: color }]} />
    </View>
  </View>
);

const MarketsIcon = ({ color, size }: { color: string; size: number }) => (
  <View style={[styles.icon, { width: size, height: size }]}>
    <View style={[styles.chartLine, { borderColor: color }]} />
    <View style={[styles.chartDot, { backgroundColor: color }]} />
  </View>
);

const TradeIcon = ({ color, size }: { color: string; size: number }) => (
  <View style={[styles.icon, { width: size, height: size }]}>
    <View style={[styles.tradeArrowUp, { borderBottomColor: color }]} />
    <View style={[styles.tradeArrowDown, { borderTopColor: color }]} />
  </View>
);

const ProfileIcon = ({ color, size }: { color: string; size: number }) => (
  <View style={[styles.icon, { width: size, height: size }]}>
    <View style={[styles.profileHead, { borderColor: color }]} />
    <View style={[styles.profileBody, { borderColor: color }]} />
  </View>
);

const styles = StyleSheet.create({
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  icon: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Portfolio icon
  portfolioBars: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: '100%',
    gap: 3,
  },
  bar: {
    width: 5,
    borderRadius: 2.5,
  },
  // Markets icon
  chartLine: {
    width: 20,
    height: 20,
    borderWidth: 2,
    borderRadius: 10,
    borderStyle: 'dashed',
  },
  chartDot: {
    position: 'absolute',
    width: 6,
    height: 6,
    borderRadius: 3,
    top: 3,
    right: 3,
  },
  // Trade icon
  tradeArrowUp: {
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderBottomWidth: 10,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
    marginBottom: 2,
  },
  tradeArrowDown: {
    width: 0,
    height: 0,
    borderLeftWidth: 8,
    borderRightWidth: 8,
    borderTopWidth: 10,
    borderLeftColor: 'transparent',
    borderRightColor: 'transparent',
  },
  // Profile icon
  profileHead: {
    width: 10,
    height: 10,
    borderWidth: 2,
    borderRadius: 5,
    marginBottom: 2,
  },
  profileBody: {
    width: 18,
    height: 10,
    borderWidth: 2,
    borderTopLeftRadius: 9,
    borderTopRightRadius: 9,
    borderBottomWidth: 0,
  },
});