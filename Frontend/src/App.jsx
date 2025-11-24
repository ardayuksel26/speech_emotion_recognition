import React from 'react';
import MainPage from './Pages/MainPage';
import { ThemeProvider } from './context/ThemeContext';

export default function App() {
  return (
    <ThemeProvider>
      <MainPage />
    </ThemeProvider>
  );
}

