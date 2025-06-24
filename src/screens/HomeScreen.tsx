// src/screens/HomeScreen.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  StatusBar,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { ElectraOrb } from '../components/common/ElectraOrb';
import { GlassCard } from '../components/common/GlassCard';
import { NeonButton } from '../components/common/NeonButton';
import { VoiceWaveform } from '../components/voice/VoiceWaveForm';
import { theme } from '../styles/theme';

export const HomeScreen: React.FC = () => {
  const [isListening, setIsListening] = useState(false);
  const [balance] = useState(12847.32);

  const handleVoicePress = () => {
    setIsListening(!isListening);
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      
      <LinearGradient
        colors={['#0A0E1A', '#131825'] as const}
        style={styles.gradient}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.greeting}>Good evening</Text>
            <Text style={styles.headerTitle}>Your Financial Assistant</Text>
          </View>

          {/* Main Orb Section */}
          <View style={styles.orbSection}>
            <TouchableOpacity onPress={handleVoicePress} activeOpacity={0.8}>
              <ElectraOrb
                size={180}
                isListening={isListening}
                isPulsing={!isListening}
              />
            </TouchableOpacity>
            
            <View style={styles.voiceSection}>
              {isListening ? (
                <>
                  <VoiceWaveform isActive={isListening} />
                  <Text style={styles.listeningText}>Listening...</Text>
                </>
              ) : (
                <>
                  <Text style={styles.voicePrompt}>
                    Tap to speak with Electra
                  </Text>
                  <Text style={styles.voiceHint}>
                    "What's my balance?" • "Pay electricity bill"
                  </Text>
                </>
              )}
            </View>
          </View>

          {/* Balance Card */}
          <GlassCard style={styles.balanceCard} variant="primary">
            <Text style={styles.balanceLabel}>Total Balance</Text>
            <Text style={styles.balanceAmount}>
              ${balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
            </Text>
            <View style={styles.balanceActions}>
              <NeonButton
                title="Send"
                onPress={() => {}}
                size="small"
                variant="outline"
                style={styles.actionButton}
              />
              <NeonButton
                title="Request"
                onPress={() => {}}
                size="small"
                variant="outline"
                style={styles.actionButton}
              />
              <NeonButton
                title="Top Up"
                onPress={() => {}}
                size="small"
                variant="primary"
                style={styles.actionButton}
              />
            </View>
          </GlassCard>

          {/* Quick Actions */}
          <View style={styles.quickActions}>
            <Text style={styles.sectionTitle}>Quick Actions</Text>
            <View style={styles.actionGrid}>
              {quickActions.map((action, index) => (
                <GlassCard key={index} style={styles.actionCard}>
                  <TouchableOpacity style={styles.actionContent}>
                    <View style={styles.actionIcon}>
                      <Text style={styles.actionEmoji}>{action.icon}</Text>
                    </View>
                    <Text style={styles.actionText}>{action.title}</Text>
                  </TouchableOpacity>
                </GlassCard>
              ))}
            </View>
          </View>
        </ScrollView>
      </LinearGradient>
    </View>
  );
};

const quickActions = [
  { icon: '💸', title: 'Transfer' },
  { icon: '📊', title: 'Analytics' },
  { icon: '💳', title: 'Cards' },
  { icon: '🎯', title: 'Goals' },
];

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0E1A',
  },
  gradient: {
    flex: 1,
  },
  scrollContent: {
    paddingBottom: 48,
  },
  header: {
    paddingHorizontal: 24,
    paddingTop: 60,
    marginBottom: 32,
  },
  greeting: {
    fontSize: 16,
    color: '#A6B0C3',
    marginBottom: 4,
  },
  headerTitle: {
    fontSize: 32,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  orbSection: {
    alignItems: 'center',
    marginBottom: 32,
  },
  voiceSection: {
    marginTop: 24,
    alignItems: 'center',
  },
  voicePrompt: {
    fontSize: 18,
    color: '#FFFFFF',
    marginBottom: 8,
  },
  voiceHint: {
    fontSize: 14,
    color: '#A6B0C3',
    textAlign: 'center',
    paddingHorizontal: 32,
  },
  listeningText: {
    fontSize: 18,
    color: '#00D4FF',
    marginTop: 16,
  },
  balanceCard: {
    marginHorizontal: 24,
    marginBottom: 32,
  },
  balanceLabel: {
    fontSize: 14,
    color: '#A6B0C3',
    marginBottom: 4,
  },
  balanceAmount: {
    fontSize: 48,
    color: '#FFFFFF',
    fontWeight: 'bold',
    marginBottom: 24,
  },
  balanceActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 4,
  },
  quickActions: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 24,
    color: '#FFFFFF',
    fontWeight: '600',
    marginBottom: 16,
  },
  actionGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
  },
  actionCard: {
    width: '50%',
    padding: 8,
  },
  actionContent: {
    alignItems: 'center',
    padding: 16,
  },
  actionIcon: {
    width: 50,
    height: 50,
    borderRadius: 25,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  actionEmoji: {
    fontSize: 24,
  },
  actionText: {
    fontSize: 14,
    color: '#FFFFFF',
  },
});