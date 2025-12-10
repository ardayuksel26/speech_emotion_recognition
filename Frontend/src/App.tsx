import React from 'react';
import MainPage from './Pages/MainPage';
import { ThemeProvider } from './context/ThemeContext';

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import AboutPage from './Pages/AboutPage';
import UseCasesPage from './Pages/UseCasesPage';
import TechnicalInfoPage from './Pages/TechnicalInfoPage';

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="flex flex-col min-h-screen">
          <Header />
          <main className="flex-grow flex flex-col">
            <Routes>
              <Route path="/" element={<MainPage />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/use-cases" element={<UseCasesPage />} />
              <Route path="/technical-info" element={<TechnicalInfoPage />} />
            </Routes>
          </main>
          <Footer />
        </div>
      </Router>
    </ThemeProvider>
  );
}

