import MainPage from './Pages/MainPage';
import { ThemeProvider } from './context/ThemeContext';

import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import AboutPage from './Pages/AboutPage';
import UseCasesPage from './Pages/UseCasesPage';
import TechnicalInfoPage from './Pages/TechnicalInfoPage';
import ExperimentalPage from './Pages/ExperimentalPage';
import ScrollToTop from './components/ScrollToTop';
import InteractiveBackground from './components/InteractiveBackground';

import { useTheme } from './context/ThemeContext';

function AppContent() {
  const { isDark } = useTheme();
  const location = useLocation();
  
  return (
    <div className={`flex flex-col min-h-screen overflow-x-hidden w-full max-w-[100vw] transition-colors duration-500 ${isDark ? "bg-[#0b0f19] text-white" : "bg-gray-50 text-slate-900"}`}>
      <InteractiveBackground />
      <Header />
      <main className="flex-grow flex flex-col pt-16 w-full">
        <ScrollToTop />
        <Routes>
          <Route path="/" element={<MainPage key={location.key} />} />
          <Route path="/experimental" element={<ExperimentalPage key={location.key} />} />
          <Route path="/about" element={<AboutPage key={location.key} />} />
          <Route path="/use-cases" element={<UseCasesPage key={location.key} />} />
          <Route path="/technical-info" element={<TechnicalInfoPage key={location.key} />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppContent />
      </Router>
    </ThemeProvider>
  );
}
