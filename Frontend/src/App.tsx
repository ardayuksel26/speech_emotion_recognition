import MainPage from './Pages/MainPage';
import { ThemeProvider } from './context/ThemeContext';

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import AboutPage from './Pages/AboutPage';
import UseCasesPage from './Pages/UseCasesPage';
import TechnicalInfoPage from './Pages/TechnicalInfoPage';
import ExperimentalPage from './Pages/ExperimentalPage';
import ScrollToTop from './components/ScrollToTop';

export default function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="flex flex-col min-h-screen overflow-x-hidden w-full max-w-[100vw]">
          <Header />
          <main className="flex-grow flex flex-col pt-16 w-full">
            <ScrollToTop />
            <Routes>
              <Route path="/" element={<MainPage />} />
              <Route path="/experimental" element={<ExperimentalPage />} />
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

