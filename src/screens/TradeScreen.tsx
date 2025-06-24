// src/screens/TradeScreen.tsx
import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Animated,
  Dimensions,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { GlassCard } from '../components/common/GlassCard';
import { NeonButton } from '../components/common/NeonButton';
import { theme } from '../styles/theme';

const { width: screenWidth } = Dimensions.get('window');

export const TradeScreen: React.FC = () => {
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [tradeAction, setTradeAction] = useState<'buy' | 'sell'>('buy');
  const [symbol, setSymbol] = useState('AAPL');
  const [quantity, setQuantity] = useState('');
  const [limitPrice, setLimitPrice] = useState('');
  const [stopPrice, setStopPrice] = useState('');
  
  const slideAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const currentPrice = 178.50;
  const estimatedCost = parseFloat(quantity || '0') * currentPrice;

  useEffect(() => {
    // Pulse animation for trade button
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.05,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);

  const quickAmounts = [10, 25, 50, 100, 500, 1000];

  const handleQuickAmount = (amount: number) => {
    setQuantity(amount.toString());
    Animated.sequence([
      Animated.timing(slideAnim, {
        toValue: 1,
        duration: 200,
        useNativeDriver: true,
      }),
      Animated.timing(slideAnim, {
        toValue: 0,
        duration: 200,
        useNativeDriver: true,
      }),
    ]).start();
  };

  const executeTrade = () => {
    // Trade execution logic here
    console.log('Executing trade:', {
      action: tradeAction,
      symbol,
      quantity,
      orderType,
      limitPrice,
      stopPrice,
    });
  };

  return (
    <KeyboardAvoidingView 
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <LinearGradient
        colors={['#000000', '#0A0A0A'] as const}
        style={styles.gradient}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Execute Trade</Text>
            <TouchableOpacity style={styles.voiceButton}>
              <Text style={styles.voiceIcon}>🎙</Text>
            </TouchableOpacity>
          </View>

          {/* Symbol Search */}
          <GlassCard style={styles.symbolCard}>
            <View style={styles.symbolHeader}>
              <TextInput
                style={styles.symbolInput}
                value={symbol}
                onChangeText={setSymbol}
                placeholder="Enter symbol"
                placeholderTextColor="#6B7280"
              />
              <View style={styles.priceInfo}>
                <Text style={styles.currentPrice}>${currentPrice}</Text>
                <Text style={[styles.priceChange, { color: theme.colors.trading.profit }]}>
                  +2.3%
                </Text>
              </View>
            </View>
          </GlassCard>

          {/* Buy/Sell Toggle */}
          <View style={styles.actionToggle}>
            <TouchableOpacity
              style={[
                styles.actionButton,
                tradeAction === 'buy' && styles.actionButtonActive,
                tradeAction === 'buy' && { backgroundColor: 'rgba(0, 255, 136, 0.1)' }
              ]}
              onPress={() => setTradeAction('buy')}
            >
              <Text style={[
                styles.actionButtonText,
                tradeAction === 'buy' && { color: theme.colors.trading.profit }
              ]}>
                Buy
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[
                styles.actionButton,
                tradeAction === 'sell' && styles.actionButtonActive,
                tradeAction === 'sell' && { backgroundColor: 'rgba(255, 51, 102, 0.1)' }
              ]}
              onPress={() => setTradeAction('sell')}
            >
              <Text style={[
                styles.actionButtonText,
                tradeAction === 'sell' && { color: theme.colors.trading.loss }
              ]}>
                Sell
              </Text>
            </TouchableOpacity>
          </View>

          {/* Order Type */}
          <View style={styles.orderTypeContainer}>
            <Text style={styles.sectionLabel}>Order Type</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {(['market', 'limit', 'stop'] as const).map((type) => (
                <TouchableOpacity
                  key={type}
                  style={[
                    styles.orderTypeButton,
                    orderType === type && styles.orderTypeActive
                  ]}
                  onPress={() => setOrderType(type)}
                >
                  <Text style={[
                    styles.orderTypeText,
                    orderType === type && styles.orderTypeTextActive
                  ]}>
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* Quantity Input */}
          <GlassCard style={styles.inputCard}>
            <Text style={styles.inputLabel}>Quantity</Text>
            <Animated.View style={{
              transform: [{
                translateX: slideAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0, 10],
                }),
              }],
            }}>
              <TextInput
                style={styles.quantityInput}
                value={quantity}
                onChangeText={setQuantity}
                placeholder="0"
                placeholderTextColor="#6B7280"
                keyboardType="numeric"
              />
            </Animated.View>
            
            {/* Quick Amount Buttons */}
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false}
              style={styles.quickAmounts}
            >
              {quickAmounts.map((amount) => (
                <TouchableOpacity
                  key={amount}
                  style={styles.quickAmountButton}
                  onPress={() => handleQuickAmount(amount)}
                >
                  <Text style={styles.quickAmountText}>{amount}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </GlassCard>

          {/* Conditional Price Inputs */}
          {orderType === 'limit' && (
            <GlassCard style={styles.inputCard}>
              <Text style={styles.inputLabel}>Limit Price</Text>
              <TextInput
                style={styles.priceInput}
                value={limitPrice}
                onChangeText={setLimitPrice}
                placeholder="0.00"
                placeholderTextColor="#6B7280"
                keyboardType="decimal-pad"
              />
            </GlassCard>
          )}

          {orderType === 'stop' && (
            <GlassCard style={styles.inputCard}>
              <Text style={styles.inputLabel}>Stop Price</Text>
              <TextInput
                style={styles.priceInput}
                value={stopPrice}
                onChangeText={setStopPrice}
                placeholder="0.00"
                placeholderTextColor="#6B7280"
                keyboardType="decimal-pad"
              />
            </GlassCard>
          )}

          {/* Order Summary */}
          <GlassCard style={styles.summaryCard} variant="primary">
            <Text style={styles.summaryTitle}>Order Summary</Text>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Symbol</Text>
              <Text style={styles.summaryValue}>{symbol}</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Quantity</Text>
              <Text style={styles.summaryValue}>{quantity || '0'} shares</Text>
            </View>
            <View style={styles.summaryRow}>
              <Text style={styles.summaryLabel}>Price</Text>
              <Text style={styles.summaryValue}>
                {orderType === 'market' ? 'Market' : 
                 orderType === 'limit' ? `$${limitPrice || '0.00'}` :
                 `$${stopPrice || '0.00'}`}
              </Text>
            </View>
            <View style={styles.summaryDivider} />
            <View style={styles.summaryRow}>
              <Text style={styles.summaryTotalLabel}>Estimated Total</Text>
              <Text style={styles.summaryTotalValue}>
                ${estimatedCost.toFixed(2)}
              </Text>
            </View>
          </GlassCard>

          {/* Execute Button */}
          <Animated.View style={{
            transform: [{ scale: pulseAnim }],
            marginHorizontal: 24,
            marginTop: 24,
          }}>
            <TouchableOpacity
              onPress={executeTrade}
              disabled={!quantity || parseFloat(quantity) === 0}
            >
              <LinearGradient
                colors={
                  tradeAction === 'buy' 
                    ? [theme.colors.trading.profit, theme.colors.trading.bull] as const
                    : [theme.colors.trading.loss, theme.colors.trading.bear] as const
                }
                style={[
                  styles.executeButton,
                  (!quantity || parseFloat(quantity) === 0) && styles.executeButtonDisabled
                ]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Text style={styles.executeButtonText}>
                  {tradeAction === 'buy' ? 'Place Buy Order' : 'Place Sell Order'}
                </Text>
                <Text style={styles.executeButtonSubtext}>
                  Swipe to confirm →
                </Text>
              </LinearGradient>
            </TouchableOpacity>
          </Animated.View>

          {/* Advanced Options */}
          <TouchableOpacity style={styles.advancedToggle}>
            <Text style={styles.advancedText}>Advanced Options</Text>
            <Text style={styles.advancedArrow}>▼</Text>
          </TouchableOpacity>
        </ScrollView>
      </LinearGradient>
    </KeyboardAvoidingView>
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
  scrollContent: {
    paddingBottom: 120,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 24,
    paddingTop: 60,
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 32,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  voiceButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.3)',
  },
  voiceIcon: {
    fontSize: 20,
  },
  symbolCard: {
    marginHorizontal: 24,
    marginBottom: 24,
  },
  symbolHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  symbolInput: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    flex: 1,
  },
  priceInfo: {
    alignItems: 'flex-end',
  },
  currentPrice: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  priceChange: {
    fontSize: 14,
    marginTop: 2,
  },
  actionToggle: {
    flexDirection: 'row',
    marginHorizontal: 24,
    marginBottom: 24,
    borderRadius: 12,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    padding: 4,
  },
  actionButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  actionButtonActive: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#6B7280',
  },
  orderTypeContainer: {
    marginBottom: 24,
  },
  sectionLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 12,
    marginLeft: 24,
  },
  orderTypeButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    marginHorizontal: 6,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.1)',
  },
  orderTypeActive: {
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    borderColor: theme.colors.primary.main,
  },
  orderTypeText: {
    color: '#6B7280',
    fontSize: 14,
  },
  orderTypeTextActive: {
    color: theme.colors.primary.main,
  },
  inputCard: {
    marginHorizontal: 24,
    marginBottom: 16,
  },
  inputLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 8,
  },
  quantityInput: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  priceInput: {
    fontSize: 24,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  quickAmounts: {
    marginTop: 8,
  },
  quickAmountButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    marginRight: 8,
    borderRadius: 16,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    borderWidth: 1,
    borderColor: 'rgba(0, 212, 255, 0.2)',
  },
  quickAmountText: {
    color: theme.colors.primary.light,
    fontSize: 14,
    fontWeight: '500',
  },
  summaryCard: {
    marginHorizontal: 24,
  },
  summaryTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  summaryLabel: {
    fontSize: 14,
    color: '#6B7280',
  },
  summaryValue: {
    fontSize: 14,
    color: '#FFFFFF',
  },
  summaryDivider: {
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    marginVertical: 12,
  },
  summaryTotalLabel: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  summaryTotalValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.colors.primary.main,
  },
  executeButton: {
    paddingVertical: 20,
    borderRadius: 16,
    alignItems: 'center',
  },
  executeButtonDisabled: {
    opacity: 0.5,
  },
  executeButtonText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#000000',
    marginBottom: 4,
  },
  executeButtonSubtext: {
    fontSize: 12,
    color: 'rgba(0, 0, 0, 0.7)',
  },
  advancedToggle: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 24,
    paddingVertical: 16,
  },
  advancedText: {
    fontSize: 14,
    color: '#6B7280',
    marginRight: 8,
  },
  advancedArrow: {
    fontSize: 12,
    color: '#6B7280',
  },
});