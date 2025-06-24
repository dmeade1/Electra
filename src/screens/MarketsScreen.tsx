// src/screens/MarketsScreen.tsx
import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Animated,
  Dimensions,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { GlassCard } from '../components/common/GlassCard';
import { theme } from '../styles/theme';

const { width: screenWidth } = Dimensions.get('window');

interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  changePercent: number;
}

interface TrendingStock {
  symbol: string;
  price: number;
  change: number;
  volume: string;
  sparkline: number[];
}

export const MarketsScreen: React.FC = () => {
  const [selectedCategory, setSelectedCategory] = useState<'indices' | 'crypto' | 'forex'>('indices');
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const slideAnim = useRef(new Animated.Value(0)).current;

  const marketIndices: MarketIndex[] = [
    { symbol: 'SPY', name: 'S&P 500', value: 452.18, change: 3.45, changePercent: 0.77 },
    { symbol: 'DIA', name: 'Dow Jones', value: 353.89, change: -1.23, changePercent: -0.35 },
    { symbol: 'QQQ', name: 'NASDAQ', value: 383.45, change: 8.92, changePercent: 2.38 },
    { symbol: 'IWM', name: 'Russell 2000', value: 198.76, change: 2.11, changePercent: 1.07 },
  ];

  const trendingStocks: TrendingStock[] = [
    { symbol: 'NVDA', price: 485.30, change: 21.45, volume: '52.3M', sparkline: [450, 460, 455, 470, 465, 480, 485] },
    { symbol: 'TSLA', price: 245.20, change: -2.95, volume: '48.7M', sparkline: [250, 248, 252, 245, 248, 246, 245] },
    { symbol: 'AAPL', price: 178.50, change: 4.12, volume: '41.2M', sparkline: [172, 174, 176, 175, 177, 178, 178.5] },
    { symbol: 'AMD', price: 122.45, change: 5.67, volume: '38.9M', sparkline: [115, 118, 120, 119, 121, 122, 122.45] },
  ];

  const cryptoMarkets = [
    { symbol: 'BTC', name: 'Bitcoin', price: 43250, change: 5.2 },
    { symbol: 'ETH', name: 'Ethereum', price: 2280, change: 3.8 },
    { symbol: 'SOL', name: 'Solana', price: 98.45, change: 12.4 },
    { symbol: 'BNB', name: 'Binance', price: 312.80, change: -1.2 },
  ];

  useEffect(() => {
    // Market pulse animation
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.1,
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

    // Slide in animation
    Animated.timing(slideAnim, {
      toValue: 1,
      duration: 500,
      useNativeDriver: true,
    }).start();
  }, []);

  const renderSparkline = (data: number[]) => {
    const max = Math.max(...data);
    const min = Math.min(...data);
    const range = max - min;
    const width = 60;
    const height = 30;

    const points = data.map((value, index) => {
      const x = (index / (data.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');

    return (
      <View style={styles.sparklineContainer}>
        <Svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
          <Path
            d={`M ${points}`}
            stroke={data[data.length - 1] > data[0] ? theme.colors.trading.profit : theme.colors.trading.loss}
            strokeWidth="2"
            fill="none"
          />
        </Svg>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#000000', '#0A0A0A'] as const}
        style={styles.gradient}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Header */}
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Markets</Text>
            <View style={styles.marketStatus}>
              <Animated.View 
                style={[
                  styles.statusDot, 
                  { 
                    backgroundColor: theme.colors.trading.profit,
                    transform: [{ scale: pulseAnim }]
                  }
                ]} 
              />
              <Text style={styles.statusText}>Live</Text>
            </View>
          </View>

          {/* Market Overview */}
          <Animated.View style={{
            opacity: slideAnim,
            transform: [{
              translateY: slideAnim.interpolate({
                inputRange: [0, 1],
                outputRange: [20, 0],
              }),
            }],
          }}>
            <ScrollView 
              horizontal 
              showsHorizontalScrollIndicator={false}
              style={styles.indicesScroll}
            >
              {marketIndices.map((index) => (
                <GlassCard key={index.symbol} style={styles.indexCard}>
                  <Text style={styles.indexSymbol}>{index.symbol}</Text>
                  <Text style={styles.indexName}>{index.name}</Text>
                  <Text style={styles.indexValue}>${index.value.toFixed(2)}</Text>
                  <View style={[
                    styles.changeContainer,
                    { backgroundColor: index.change > 0 ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 51, 102, 0.1)' }
                  ]}>
                    <Text style={[
                      styles.changeText,
                      { color: index.change > 0 ? theme.colors.trading.profit : theme.colors.trading.loss }
                    ]}>
                      {index.change > 0 ? '+' : ''}{index.changePercent.toFixed(2)}%
                    </Text>
                  </View>
                </GlassCard>
              ))}
            </ScrollView>
          </Animated.View>

          {/* Category Tabs */}
          <View style={styles.categoryTabs}>
            {(['indices', 'crypto', 'forex'] as const).map((category) => (
              <TouchableOpacity
                key={category}
                style={[
                  styles.categoryTab,
                  selectedCategory === category && styles.categoryTabActive
                ]}
                onPress={() => setSelectedCategory(category)}
              >
                <Text style={[
                  styles.categoryTabText,
                  selectedCategory === category && styles.categoryTabTextActive
                ]}>
                  {category.charAt(0).toUpperCase() + category.slice(1)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          {/* Trending Stocks */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>🔥 Trending Now</Text>
            {trendingStocks.map((stock) => (
              <TouchableOpacity key={stock.symbol}>
                <GlassCard style={styles.stockCard}>
                  <View style={styles.stockLeft}>
                    <Text style={styles.stockSymbol}>{stock.symbol}</Text>
                    <Text style={styles.stockVolume}>Vol: {stock.volume}</Text>
                  </View>
                  <View style={styles.stockCenter}>
                    {renderSparkline(stock.sparkline)}
                  </View>
                  <View style={styles.stockRight}>
                    <Text style={styles.stockPrice}>${stock.price.toFixed(2)}</Text>
                    <Text style={[
                      styles.stockChange,
                      { color: stock.change > 0 ? theme.colors.trading.profit : theme.colors.trading.loss }
                    ]}>
                      {stock.change > 0 ? '+' : ''}${stock.change.toFixed(2)}
                    </Text>
                  </View>
                </GlassCard>
              </TouchableOpacity>
            ))}
          </View>

          {/* Crypto Section */}
          {selectedCategory === 'crypto' && (
            <View style={styles.section}>
              <Text style={styles.sectionTitle}>Cryptocurrency</Text>
              {cryptoMarkets.map((crypto) => (
                <GlassCard key={crypto.symbol} style={styles.cryptoCard}>
                  <View style={styles.cryptoLeft}>
                    <Text style={styles.cryptoSymbol}>{crypto.symbol}</Text>
                    <Text style={styles.cryptoName}>{crypto.name}</Text>
                  </View>
                  <View style={styles.cryptoRight}>
                    <Text style={styles.cryptoPrice}>
                      ${crypto.price.toLocaleString()}
                    </Text>
                    <View style={[
                      styles.cryptoBadge,
                      { backgroundColor: crypto.change > 0 ? 'rgba(0, 255, 136, 0.1)' : 'rgba(255, 51, 102, 0.1)' }
                    ]}>
                      <Text style={[
                        styles.cryptoChange,
                        { color: crypto.change > 0 ? theme.colors.trading.profit : theme.colors.trading.loss }
                      ]}>
                        {crypto.change > 0 ? '+' : ''}{crypto.change}%
                      </Text>
                    </View>
                  </View>
                </GlassCard>
              ))}
            </View>
          )}

          {/* Market Movers */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>📈 Top Gainers</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {[
                { symbol: 'GME', change: '+18.5%' },
                { symbol: 'AMC', change: '+12.3%' },
                { symbol: 'RIOT', change: '+9.8%' },
                { symbol: 'MARA', change: '+8.2%' },
              ].map((mover) => (
                <View key={mover.symbol} style={styles.moverCard}>
                  <Text style={styles.moverSymbol}>{mover.symbol}</Text>
                  <Text style={[styles.moverChange, { color: theme.colors.trading.profit }]}>
                    {mover.change}
                  </Text>
                </View>
              ))}
            </ScrollView>
          </View>

          {/* Market News Flash */}
          <GlassCard style={styles.newsCard} variant="primary">
            <Text style={styles.newsTitle}>📰 Market Flash</Text>
            <Text style={styles.newsText}>
              Fed maintains rates • Tech earnings beat expectations • Oil prices surge on supply concerns
            </Text>
          </GlassCard>
        </ScrollView>
      </LinearGradient>
    </View>
  );
};

// Add this at the top of the file after imports
const Svg = ({ children, ...props }: any) => (
  <View {...props}>{children}</View>
);

const Path = ({ d, stroke, strokeWidth, fill }: any) => (
  <View style={{ backgroundColor: stroke, height: strokeWidth }} />
);

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
  marketStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 255, 136, 0.1)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 6,
  },
  statusText: {
    fontSize: 12,
    color: theme.colors.trading.profit,
    fontWeight: '600',
  },
  indicesScroll: {
    marginBottom: 24,
    paddingLeft: 24,
  },
  indexCard: {
    marginRight: 12,
    width: 140,
    alignItems: 'center',
    padding: 16,
  },
  indexSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  indexName: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 8,
  },
  indexValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  changeContainer: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  changeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  categoryTabs: {
    flexDirection: 'row',
    marginHorizontal: 24,
    marginBottom: 24,
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
    borderRadius: 12,
    padding: 4,
  },
  categoryTab: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 8,
  },
  categoryTabActive: {
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
  },
  categoryTabText: {
    fontSize: 14,
    color: '#6B7280',
    fontWeight: '500',
  },
  categoryTabTextActive: {
    color: theme.colors.primary.main,
  },
  section: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  stockCard: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    padding: 16,
  },
  stockLeft: {
    flex: 1,
  },
  stockSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  stockVolume: {
    fontSize: 12,
    color: '#6B7280',
    marginTop: 2,
  },
  stockCenter: {
    marginHorizontal: 16,
  },
  sparklineContainer: {
    width: 60,
    height: 30,
  },
  stockRight: {
    alignItems: 'flex-end',
  },
  stockPrice: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  stockChange: {
    fontSize: 14,
    marginTop: 2,
  },
  cryptoCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
    padding: 16,
  },
  cryptoLeft: {
    flex: 1,
  },
  cryptoSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  cryptoName: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 2,
  },
  cryptoRight: {
    alignItems: 'flex-end',
  },
  cryptoPrice: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  cryptoBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  cryptoChange: {
    fontSize: 12,
    fontWeight: '600',
  },
  moverCard: {
    backgroundColor: 'rgba(0, 255, 136, 0.05)',
    borderWidth: 1,
    borderColor: 'rgba(0, 255, 136, 0.2)',
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    marginRight: 12,
    alignItems: 'center',
  },
  moverSymbol: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  moverChange: {
    fontSize: 16,
    fontWeight: '600',
  },
  newsCard: {
    marginHorizontal: 24,
    marginBottom: 24,
  },
  newsTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 8,
  },
  newsText: {
    fontSize: 14,
    color: '#A6B0C3',
    lineHeight: 20,
  },
});