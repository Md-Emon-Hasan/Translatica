import { MainLayout } from './components/layout/MainLayout';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { TranslatorCard } from './features/translator/TranslatorCard';
import { Features } from './components/ui/Features';
import { useParticles } from './hooks/useParticles';

function App() {
  const particlesRef = useParticles();

  return (
    <MainLayout>
      <div ref={particlesRef} className="fixed inset-0 pointer-events-none z-[1] overflow-hidden" />

      <div className="w-full relative z-[2] flex flex-col items-center">
        <Header />
        <TranslatorCard />
        <Features />
      </div>

      <Footer />
    </MainLayout>
  );
}

export default App;
