// App.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { VoiceScreen } from '/Users/andrewmeade/Desktop/Electra/src/screens/VoiceScreen';
import { TabNavigator } from '/Users/andrewmeade/Desktop/Electra/src/components/navigation/TabNavigator';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        screenOptions={{
          headerShown: false,
          cardStyle: {
            backgroundColor: '#000000',
          },
        }}
      >
        <Stack.Screen name="Voice" component={VoiceScreen} />
        <Stack.Screen name="MainTabs" component={TabNavigator} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}