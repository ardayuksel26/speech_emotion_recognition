import Header from "../components/Header";
import Hero from "../components/Hero";
import Footer from "../components/Footer";

const MainPage = () => {
  return (
    <div className="min-h-screen w-full flex flex-col bg-gray-100 overflow-x-hidden">
      <Header />

      {/* Hero’nun kalan alanı doldurması için main’e flex veriyoruz */}
      <main className="flex-1 flex">
        <Hero />
      </main>

      <Footer />
    </div>
  );
};

export default MainPage;
