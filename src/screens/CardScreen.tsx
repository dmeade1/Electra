// src/screens/CardsScreen.tsx
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { GlassCard } from '../components/common/GlassCard';
import { NeonButton } from '../components/common/NeonButton';

const { width } = Dimensions.get('window');

export const CardsScreen: React.FC = () => {
  const [selectedCard, setSelectedCard] = useState(0);

  const cards = [
    {
      id: 0,
      type: 'Virtual',
      number: '•••• •••• •••• 4532',
      name: 'ANDREW MEADE',
      expiry: '12/26',
      balance: 12847.32,
      gradient: ['#00D4FF', '#0099CC'],
    },
    {
      id: 1,
      type: 'Physical',
      number: '•••• •••• •••• 7891',
      name: 'ANDREW MEADE',
      expiry: '08/25',
      balance: 5230.00,
      gradient: ['#8B5CF6', '#6D28D9'],
    },
  ];

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#0A0E1A', '#131825'] as const}
        style={styles.gradient}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <Text style={styles.headerTitle}>My Cards</Text>
            <Text style={styles.headerSubtitle}>Manage your payment methods</Text>
          </View>

          {/* Card Carousel */}
          <ScrollView
            horizontal
            pagingEnabled
            showsHorizontalScrollIndicator={false}
            onMomentumScrollEnd={(e) => {
              const newIndex = Math.round(e.nativeEvent.contentOffset.x / (width - 48));
              setSelectedCard(newIndex);
            }}
            style={styles.cardCarousel}
          >
            {cards.map((card) => (
              <View key={card.id} style={styles.cardWrapper}>
                <LinearGradient
                  colors={[card.gradient[0], card.gradient[1]] as const}
                  style={styles.card}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                >
                  <View style={styles.cardHeader}>
                    <Text style={styles.cardType}>{card.type} Card</Text>
                    <Text style={styles.cardLogo}>Electra</Text>
                  </View>
                  
                  <View style={styles.cardChip} />
                  
                  <Text style={styles.cardNumber}>{card.number}</Text>
                  
                  <View style={styles.cardFooter}>
                    <View>
                      <Text style={styles.cardLabel}>CARD HOLDER</Text>
                      <Text style={styles.cardName}>{card.name}</Text>
                    </View>
                    <View>
                      <Text style={styles.cardLabel}>EXPIRES</Text>
                      <Text style={styles.cardExpiry}>{card.expiry}</Text>
                    </View>
                  </View>
                </LinearGradient>
              </View>
            ))}
          </ScrollView>

          {/* Card Indicators */}
          <View style={styles.indicators}>
            {cards.map((_, index) => (
              <View
                key={index}
                style={[
                  styles.indicator,
                  selectedCard === index && styles.indicatorActive,
                ]}
              />
            ))}
          </View>

          {/* Quick Actions */}
          <View style={styles.quickActions}>
            <NeonButton
              title="Freeze Card"
              onPress={() => {}}
              variant="outline"
              style={styles.actionButton}
            />
            <NeonButton
              title="Card Details"
              onPress={() => {}}
              variant="primary"
              style={styles.actionButton}
            />
          </View>

          {/* Recent Transactions */}
          <GlassCard style={styles.transactionsCard}>
            <Text style={styles.sectionTitle}>Recent Transactions</Text>
            {recentTransactions.map((transaction, index) => (
              <View
                key={index}
                style={[
                  styles.transaction,
                  index < recentTransactions.length - 1 && styles.transactionBorder,
                ]}
              >
                <View style={styles.transactionLeft}>
                  <View style={styles.transactionIcon}>
                    <Text>{transaction.icon}</Text>
                  </View>
                  <View>
                    <Text style={styles.transactionTitle}>{transaction.title}</Text>
                    <Text style={styles.transactionDate}>{transaction.date}</Text>
                  </View>
                </View>
                <Text style={styles.transactionAmount}>
                  ${Math.abs(transaction.amount).toFixed(2)}
                </Text>
              </View>
            ))}
          </GlassCard>

          {/* Card Settings */}
          <View style={styles.settings}>
            <TouchableOpacity style={styles.settingItem}>
              <Text style={styles.settingText}>Spending Limits</Text>
              <Text style={styles.settingArrow}>›</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.settingItem}>
              <Text style={styles.settingText}>Online Payments</Text>
              <View style={styles.toggle}>
                <View style={styles.toggleActive} />
              </View>
            </TouchableOpacity>
            <TouchableOpacity style={styles.settingItem}>
              <Text style={styles.settingText}>Contactless Payments</Text>
              <View style={styles.toggle}>
                <View style={styles.toggleActive} />
              </View>
            </TouchableOpacity>
          </View>
        </ScrollView>
      </LinearGradient>
    </View>
  );
};

const recentTransactions = [
  { icon: '🛒', title: 'Walmart', date: 'Today, 3:42 PM', amount: -67.50 },
  { icon: '⛽', title: 'Shell Gas Station', date: 'Today, 11:20 AM', amount: -45.00 },
  { icon: '🍔', title: 'McDonald\'s', date: 'Yesterday', amount: -12.30 },
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
    paddingBottom: 120,
  },
  header: {
    paddingHorizontal: 24,
    paddingTop: 60,
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 32,
    color: '#FFFFFF',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#A6B0C3',
  },
  cardCarousel: {
    marginBottom: 20,
  },
  cardWrapper: {
    width: width - 48,
    paddingHorizontal: 12,
  },
  card: {
    height: 200,
    borderRadius: 20,
    padding: 24,
    justifyContent: 'space-between',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  cardType: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
  },
  cardLogo: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
  },
  cardChip: {
    width: 40,
    height: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    borderRadius: 6,
  },
  cardNumber: {
    color: '#FFFFFF',
    fontSize: 18,
    letterSpacing: 2,
  },
  cardFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  cardLabel: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: 10,
    marginBottom: 4,
  },
  cardName: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  cardExpiry: {
    color: '#FFFFFF',
    fontSize: 14,
  },
  indicators: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginBottom: 24,
  },
  indicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    marginHorizontal: 4,
  },
  indicatorActive: {
    backgroundColor: '#00D4FF',
    width: 24,
  },
  quickActions: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  actionButton: {
    flex: 1,
    marginHorizontal: 6,
  },
  transactionsCard: {
    marginHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 18,
    color: '#FFFFFF',
    fontWeight: '600',
    marginBottom: 16,
  },
  transaction: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
  },
  transactionBorder: {
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
  },
  transactionLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  transactionIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 12,
  },
  transactionTitle: {
    fontSize: 16,
    color: '#FFFFFF',
    marginBottom: 2,
  },
  transactionDate: {
    fontSize: 14,
    color: '#6B7280',
  },
  transactionAmount: {
    fontSize: 16,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  settings: {
    paddingHorizontal: 24,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
  },
  settingText: {
    fontSize: 16,
    color: '#FFFFFF',
  },
  settingArrow: {
    fontSize: 24,
    color: '#6B7280',
  },
  toggle: {
    width: 48,
    height: 28,
    backgroundColor: 'rgba(0, 212, 255, 0.2)',
    borderRadius: 14,
    padding: 2,
    justifyContent: 'center',
  },
  toggleActive: {
    width: 24,
    height: 24,
    backgroundColor: '#00D4FF',
    borderRadius: 12,
    alignSelf: 'flex-end',
  },
});