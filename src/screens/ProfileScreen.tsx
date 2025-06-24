// src/screens/ProfileScreen.tsx
import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { GlassCard } from '../components/common/GlassCard';
import { NeonButton } from '../components/common/NeonButton';

export const ProfileScreen: React.FC = () => {
  const menuItems = [
    { icon: '👤', title: 'Personal Information', subtitle: 'Update your details' },
    { icon: '🔔', title: 'Notifications', subtitle: 'Manage alerts', badge: '3' },
    { icon: '🔒', title: 'Security', subtitle: 'Password & biometrics' },
    { icon: '💳', title: 'Payment Methods', subtitle: 'Cards & accounts' },
    { icon: '📊', title: 'Spending Limits', subtitle: 'Set your budgets' },
    { icon: '🎯', title: 'Financial Goals', subtitle: 'Track your progress' },
    { icon: '📄', title: 'Documents', subtitle: 'Statements & reports' },
    { icon: '❓', title: 'Help & Support', subtitle: 'Get assistance' },
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
          {/* Profile Header */}
          <View style={styles.header}>
            <View style={styles.profileInfo}>
              <LinearGradient
                colors={['#00D4FF', '#0099CC'] as const}
                style={styles.avatar}
              >
                <Text style={styles.avatarText}>AM</Text>
              </LinearGradient>
              <View style={styles.userInfo}>
                <Text style={styles.userName}>Andrew Meade</Text>
                <Text style={styles.userEmail}>andrew@electra.finance</Text>
                <TouchableOpacity style={styles.premiumBadge}>
                  <LinearGradient
                    colors={['#FFD700', '#FFA500'] as const}
                    style={styles.premiumGradient}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                  >
                    <Text style={styles.premiumText}>✨ Premium Member</Text>
                  </LinearGradient>
                </TouchableOpacity>
              </View>
            </View>
          </View>

          {/* Account Summary */}
          <GlassCard style={styles.summaryCard} variant="primary">
            <Text style={styles.summaryTitle}>Account Summary</Text>
            <View style={styles.summaryGrid}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Member Since</Text>
                <Text style={styles.summaryValue}>Jan 2024</Text>
              </View>
              <View style={styles.summaryDivider} />
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Total Saved</Text>
                <Text style={styles.summaryValue}>$15,420</Text>
              </View>
              <View style={styles.summaryDivider} />
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>Goals Met</Text>
                <Text style={styles.summaryValue}>8/10</Text>
              </View>
            </View>
          </GlassCard>

          {/* Menu Items */}
          <View style={styles.menuSection}>
            {menuItems.map((item, index) => (
              <TouchableOpacity key={index} style={styles.menuItem}>
                <GlassCard style={styles.menuCard}>
                  <View style={styles.menuContent}>
                    <View style={styles.menuLeft}>
                      <View style={styles.menuIcon}>
                        <Text style={styles.menuEmoji}>{item.icon}</Text>
                      </View>
                      <View>
                        <Text style={styles.menuTitle}>{item.title}</Text>
                        <Text style={styles.menuSubtitle}>{item.subtitle}</Text>
                      </View>
                    </View>
                    <View style={styles.menuRight}>
                      {item.badge && (
                        <View style={styles.badge}>
                          <Text style={styles.badgeText}>{item.badge}</Text>
                        </View>
                      )}
                      <Text style={styles.menuArrow}>›</Text>
                    </View>
                  </View>
                </GlassCard>
              </TouchableOpacity>
            ))}
          </View>

          {/* App Settings */}
          <View style={styles.settingsSection}>
            <Text style={styles.sectionTitle}>App Settings</Text>
            <GlassCard>
              <TouchableOpacity style={styles.settingItem}>
                <Text style={styles.settingText}>Dark Mode</Text>
                <View style={styles.toggle}>
                  <View style={styles.toggleActive} />
                </View>
              </TouchableOpacity>
              <TouchableOpacity style={[styles.settingItem, styles.settingBorder]}>
                <Text style={styles.settingText}>Face ID</Text>
                <View style={styles.toggle}>
                  <View style={styles.toggleActive} />
                </View>
              </TouchableOpacity>
              <TouchableOpacity style={styles.settingItem}>
                <Text style={styles.settingText}>Push Notifications</Text>
                <View style={styles.toggleInactive}>
                  <View style={styles.toggleOff} />
                </View>
              </TouchableOpacity>
            </GlassCard>
          </View>

          {/* Sign Out Button */}
          <View style={styles.signOutSection}>
            <NeonButton
              title="Sign Out"
              onPress={() => {}}
              variant="outline"
              size="large"
            />
            <Text style={styles.versionText}>Electra v1.0.0</Text>
          </View>
        </ScrollView>
      </LinearGradient>
    </View>
  );
};

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
  profileInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  avatarText: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  userInfo: {
    flex: 1,
  },
  userName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
    marginBottom: 4,
  },
  userEmail: {
    fontSize: 16,
    color: '#A6B0C3',
    marginBottom: 8,
  },
  premiumBadge: {
    alignSelf: 'flex-start',
  },
  premiumGradient: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  premiumText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#1A1A1A',
  },
  summaryCard: {
    marginHorizontal: 24,
    marginBottom: 32,
  },
  summaryTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 20,
  },
  summaryGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  summaryItem: {
    flex: 1,
    alignItems: 'center',
  },
  summaryLabel: {
    fontSize: 12,
    color: '#A6B0C3',
    marginBottom: 4,
  },
  summaryValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  summaryDivider: {
    width: 1,
    height: 30,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
  },
  menuSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  menuItem: {
    marginBottom: 12,
  },
  menuCard: {
    padding: 0,
  },
  menuContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
  },
  menuLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  menuIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(0, 212, 255, 0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 16,
  },
  menuEmoji: {
    fontSize: 20,
  },
  menuTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 2,
  },
  menuSubtitle: {
    fontSize: 14,
    color: '#6B7280',
  },
  menuRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  badge: {
    backgroundColor: '#EF4444',
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 2,
    marginRight: 8,
  },
  badgeText: {
    fontSize: 12,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  menuArrow: {
    fontSize: 24,
    color: '#6B7280',
  },
  settingsSection: {
    paddingHorizontal: 24,
    marginBottom: 32,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FFFFFF',
    marginBottom: 16,
  },
  settingItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 16,
  },
  settingBorder: {
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255, 255, 255, 0.05)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(255, 255, 255, 0.05)',
  },
  settingText: {
    fontSize: 16,
    color: '#FFFFFF',
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
  toggleInactive: {
    width: 48,
    height: 28,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 14,
    padding: 2,
    justifyContent: 'center',
  },
  toggleOff: {
    width: 24,
    height: 24,
    backgroundColor: '#6B7280',
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  signOutSection: {
    paddingHorizontal: 24,
    alignItems: 'center',
  },
  versionText: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 16,
  },
});